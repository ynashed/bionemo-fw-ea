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
from torch.nn.modules.loss import _WeightedLoss
from typing import List

class SSPredLoss(_WeightedLoss):
    def __init__(self, **kwargs):
        super(SSPredLoss, self).__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input, target, masks):
        assert len(input) == len(target)
        loss = 0
        for i in range(len(input)):
            masked_out = mask_tensor(masks[i], input[i]).permute(0, 2, 1)
            cur_loss = self.loss_fn(masked_out, target[i].permute(0, 2, 1))
            loss += cur_loss
        return loss

#Network architecture for secondary structure prediction. { display-mode: "form" }
#Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet(torch.nn.Module):
    def __init__(self, embed_dim: int, output_sizes: List[int]):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d(embed_dim, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.25),
                        )
        n_final_in = 32
        self.class_heads = torch.nn.ModuleList([])
        for head_size in output_sizes:
            self.class_heads.append(torch.nn.Conv2d(n_final_in, head_size, kernel_size=(7,1), padding=(3,0)))

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        outputs = []
        for head in self.class_heads:
            output = head(x)
            outputs.append(output.squeeze(dim=-1).permute(0,2,1)) # OUT: (B x L x output_size)
        return outputs 


def prepare_batch(batch, label_names):
    num_labels = len(label_names)
    max_batch_seq_len = batch["seq_len"].max()
    embeddings = batch["embeddings"][:, :max_batch_seq_len, :]
    labels = []
    masks = []
    for i in range(num_labels):
        cur_label = batch[label_names[i]][:, :max_batch_seq_len, :]
        labels.append(cur_label.to("cuda"))
        mask = batch["_".join(["mask", label_names[i]])][:, :max_batch_seq_len].to("cuda")
        masks.append(mask)
    return embeddings.to("cuda"), labels, masks

def mask_tensor(mask, tensor):
    dims = tensor.size()
    mask = torch.repeat_interleave(mask, dims[2]).reshape(dims[0], -1, dims[2]).to("cuda")
    output_tensor = torch.mul(mask, tensor)

    return output_tensor

def train_one_epoch(model, traindata, training_loader, loss_fn, optimizer, scheduler):
    running_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels, masks = prepare_batch(data, 
                                             traindata.label_names
                                             )
        # Make predictions for this batch
        torch.set_grad_enabled(True)
        outputs = model(inputs)
        # Compute the loss and its gradients

        # We only compute loss for resolved components. So multiply all outputs with labels3 before
        # CrossEntropyLoss expected input dims: B X C X L : Batch X Classes X SequenceLength
        # Output from Model : B X L X C : Batch X SequenceLength X Classes
        # Permute the outputs and labels to CrossEntropyLoss expected dimension order
        loss = loss_fn(outputs, labels, masks)
 
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
    loss_fn = SSPredLoss()
    optimizer, scheduler = setup_optim(cfg, model)
    
    for epoch in range(cfg.num_epochs):
        model.train(True)
        logging.info('EPOCH {}'.format(epoch+1))
        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = train_one_epoch(model, traindata, train_dataloader, loss_fn, optimizer, scheduler)
        logging.info('Training Avg Loss: {}'.format(avg_loss))
       
def test(model, data, dataloader):
    loss_fn = SSPredLoss()
    running_vloss = 0.0
    avg_vloss = 0.0
    avg_acc = [0.0] * len(data.labels_sizes)
    for n, vdata in enumerate(tqdm(dataloader)):
        vinputs, vlabels, mask = prepare_batch(vdata, data.label_names)
        voutputs = model(vinputs)
        # We only compute loss for resolved components. So multiply all outputs with labels3 before
        vloss = loss_fn(voutputs, vlabels, mask)
        running_vloss += vloss

        for i in range(len(avg_acc)):
            avg_acc[i] += (sec_str_pred_accuracy(voutputs, vlabels, i))
        
    avg_vloss = running_vloss / (n + 1)
    for i in range(len(avg_acc)):
        avg_acc[i] = avg_acc[i] / (n + 1)
    return avg_vloss, avg_acc

def sec_str_pred_accuracy(outputs: List, targets: List, label_id: int):
    output = outputs[label_id]
    target = targets[label_id]
    seq_len = list(target.sum(axis=2).sum(axis=1).cpu().numpy().astype("int"))
    target_seq = target.argmax(2)
    pred_seq = output.argmax(2)
    acc = [calculate_accuracy(pred[:l], label[:l]) for (l, pred, label) in zip(seq_len, pred_seq, target_seq)]
    return torch.tensor(np.mean(acc), device="cuda")


def main(cfg, traindata, train_dataloader, testdata, test_dataloader) -> None:

    pretrain_model_hidden_size = traindata.get_hidden_size()
    model = ConvNet(pretrain_model_hidden_size, output_sizes=traindata.labels_sizes).to("cuda")
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

