# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from nemo.core.optim.optimizers import get_optimizer
from nemo.core.optim.lr_scheduler import get_scheduler
from omegaconf import OmegaConf
from nemo.utils import logging
from bionemo.model.protein.downstream.sec_str_pred_data import *
from nemo.utils import logging


#Network architecture for secondary structure prediction. { display-mode: "form" }
#Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet(torch.nn.Module):
    def __init__(self, embed_dim):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d(embed_dim, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.25),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Conv2d(n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
        self.dssp8_classifier = torch.nn.Conv2d(n_final_in, 8, kernel_size=(7,1), padding=(3,0))
        self.diso_classifier = torch.nn.Conv2d(n_final_in, 2, kernel_size=(7,1), padding=(3,0))       

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat


def prepare_batch(batch):
    max_batch_seq_len = batch["emb_size"].max()
    embeddings = batch["embeddings"][:, :max_batch_seq_len, :]
    labels1 = batch["3state"][:, :max_batch_seq_len, :]
    labels2 = batch["8state"][:, :max_batch_seq_len, :]
    labels3 = batch["2state"][:, :max_batch_seq_len, :]
    return embeddings.to("cuda"), labels1.to("cuda"), labels2.to("cuda"), labels3.to("cuda")

def mask_tensor(mask_list, tensor):
    dims = tensor.size()
    mask_tensor = torch.tensor(np.array(mask_list).repeat((dims[2])).reshape(dims[0], -1, dims[2])).to("cuda")
    output_tensor = torch.mul(mask_tensor, tensor)

    return output_tensor

def train_one_epoch(model, traindata, training_loader, loss_fn, optimizer, scheduler):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels1, labels2, labels3 = prepare_batch(data)
        # Make predictions for this batch
        torch.set_grad_enabled(True)
        outputs = model(inputs)
        # Compute the loss and its gradients

        # We only compute loss for resolved components. So multiply all outputs with labels3 before
        # CrossEntropyLoss expected input dims: B X C X L : Batch X Classes X SequenceLength
        # Output from Model : B X L X C : Batch X SequenceLength X Classes
        # Permute the outputs and labels to CrossEntropyLoss expected dimension order
        mask_list = traindata.tensor2list(labels3)
        mask_list = np.array(mask_list).reshape(labels3.size()[:2]).tolist()
        masked_output0 = mask_tensor(mask_list, outputs[0]).permute(0,2,1)
        masked_output1 = mask_tensor(mask_list, outputs[1]).permute(0,2,1)
        masked_output2 = outputs[2].permute(0,2,1)
        loss1 = loss_fn(masked_output0, labels1.permute(0,2,1))
        loss2 = loss_fn(masked_output1, labels2.permute(0,2,1))
        loss3 = loss_fn(masked_output2, labels3.permute(0,2,1))
        loss = loss1 + loss2 + loss3
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        scheduler.step()

        # Gather loss
        running_loss += loss.item()

    return running_loss / (i + 1) 

def calculate_accuracy(predicted, labels):

    matches = [int(p==l) for p, l in zip(predicted, labels)]
    num_correct = sum(matches)
    accuracy = 100 * num_correct / len(labels)
    return accuracy


def setup_optim(cfg, model):
    params = OmegaConf.to_container(cfg)
    optim_params = params["optim"]
    opt_name = optim_params.pop("name")
    sched_params = optim_params.pop("sched")
    sched_name = sched_params.pop("name")
    opt_cls = get_optimizer(opt_name)
    optimizer = opt_cls(model.parameters(), **optim_params)
    sched_cls = get_scheduler(sched_name)
    scheduler = sched_cls(optimizer, **sched_params)
    return optimizer, scheduler

def train(cfg, model, traindata, train_dataloader):
    """Train and Save the model."""
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optim(cfg, model)
    
    for epoch in range(cfg.num_epochs):
        model.train(True)
        logging.info('EPOCH {}'.format(epoch+1))
        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = train_one_epoch(model, traindata, train_dataloader, loss_fn, optimizer, scheduler)
        logging.info('Training Avg Loss: {}'.format(avg_loss))
       
def test(model, data, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    running_vloss = 0.0
    avg_vloss = 0.0
    avg_three_state_acc = 0.0
    avg_eight_state_acc = 0.0
    avg_two_state_acc = 0.0
    for i, vdata in enumerate(tqdm(dataloader)):
        vinputs, vlabels1, vlabels2, vlabels3 = prepare_batch(vdata)
        voutputs = model(vinputs)
        # We only compute loss for resolved components. So multiply all outputs with labels3 before
        mask_list = data.tensor2list(vlabels3)
        masked_output0 = mask_tensor(mask_list, voutputs[0]).permute(0,2,1)
        masked_output1 = mask_tensor(mask_list, voutputs[1]).permute(0,2,1)
        masked_output2 = mask_tensor(mask_list, voutputs[2]).permute(0,2,1)
        vloss1 = loss_fn(masked_output0, vlabels1.permute(0,2,1))
        vloss2 = loss_fn(masked_output1, vlabels2.permute(0,2,1))
        vloss3 = loss_fn(masked_output2, vlabels3.permute(0,2,1))
        vloss = vloss1 + vloss2 + vloss3
        running_vloss += vloss

        label_three_state_seq = data.num2label(vlabels1, "three_state")
        pred_three_state_seq = data.num2label(voutputs[0], "three_state")
        label_eight_state_seq = data.num2label(vlabels2, "eight_state")
        pred_eight_state_seq = data.num2label(voutputs[1], "eight_state")

        label_two_state_seq = data.tensor2list(vlabels3)
        label_two_state_seq = [str(x) for x in label_two_state_seq]
        pred_two_state_seq = data.tensor2list(voutputs[2])
        pred_two_state_seq = [str(x) for x in pred_two_state_seq]

        avg_three_state_acc += calculate_accuracy(pred_three_state_seq, label_three_state_seq)
        avg_eight_state_acc += calculate_accuracy(pred_eight_state_seq, label_eight_state_seq)
        avg_two_state_acc += calculate_accuracy(pred_two_state_seq, label_two_state_seq)

    avg_vloss = running_vloss / (i + 1)
    avg_three_state_acc = avg_three_state_acc / (i + 1)
    avg_eight_state_acc = avg_eight_state_acc / (i + 1)
    avg_two_state_acc = avg_two_state_acc / (i + 1)
    avg_acc = [avg_three_state_acc, avg_eight_state_acc, avg_two_state_acc]
    return avg_vloss, avg_acc


def main(cfg, traindata, train_dataloader, testdata, test_dataloader) -> None:

    pretrain_model_hidden_size = traindata.get_hidden_size()
    model = ConvNet(pretrain_model_hidden_size).to("cuda")
    logging.info("Starting Secondary Structure Training...")
    train(cfg, model, traindata, train_dataloader)
    
    pretrain_model_hidden_size = testdata.get_hidden_size()
    avg_loss, avg_acc = test(model, testdata, test_dataloader)
    logging.info('Total Average Test Loss {}'.format(avg_loss))
    logging.info('Test Data Three State Accuracy {}'.format(avg_acc[0]))
    logging.info('Test Data Eight State Accuracy {}'.format(avg_acc[1]))
    logging.info('Test Data Two State Accuracy {}'.format(avg_acc[2]))
    result_dict = {}
    result_dict["3-state_accuracy"] = avg_acc[0]
    result_dict["8-state_accuracy"] = avg_acc[1]
    result_dict["ss_test_loss"] = avg_loss
    return result_dict


if __name__ == '__main__':
    main()

