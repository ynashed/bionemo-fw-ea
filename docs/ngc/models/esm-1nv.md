# Model Overview

ESM-1nv is a model that has been trained on protein sequences. The embeddings from its encoder can be used as features for predictive models. 

## Model Architecture

The model uses an architecture called Bidirectional Encoder Representations from Transformers (BERT) and is based on the ESM-1 model [1, 2]. Pre-norm layer normalization and GELU activation are used throughout. The model has 6 layers, 12 attention heads, and a hidden space dimension of 768. Dropout was set to 0.1 during training.

## Training

ESM-1nv was developed using the NeMo framework. It was trained with data parallelism on 176 A100 GPUs for 420 epochs (approximately 349500 iterations) using a micro batch size of 370 sequences per GPU. Cosine annealing was used, with a minimum learning rate of 2.0e-05, 500 warmup steps and 50000 constant steps. Fused Adam optimization was used with parameters beta1 = 0.9, beta2 = 0.98, and weight decay = 0.01. Dropout was set to 0.1 during training. The model training was then continued on 144 A100 GPUs for an additional 600 epochs, resulting in total of 957610 iterations. The weights of the last 47 checkpoints were averaged to produce the final model.

Other training parameters can be found [here](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/examples/protein/esm1nv/conf/pretrain_small.yaml).

The training procedure entailed randomly masking  15% of the input tokens, and minimizing a cross-entropy categorical loss in order to predict the masked tokens [2].

## Dataset

UniRef50 (release 05/2022) was used for training [3]. The reference sequence for each cluster was selected, resulting in approximately 52M protein sequences. The sequences were randomly split with 5K sequences in validation, 1M sequences in test, and the remaining in train. Truncation of protein sequences longer than 1024 amino acids and data masking was performed as described previously [2].

## How to Use this Model

The [Quickstart guide](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/QUICKSTART.md) has instructions for pre-training, and an example of how to use a pretrained model for inference can be found in [this notebook](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/examples/protein/esm1nv/nbs/Inference.ipynb).

## References

1. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, L., Ma, J., Fergus, R., "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences", _Proc. National. Acad. Sci._, *118*, e2016239118 (2021).
2. Devlin, J., Chang, M. W., Lee, K. & Toutanova, K., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", _ArXiv_ (2018).
3. The Uniprot Consortium, Bateman, A., Marin, M. J., Orchard, S., Magrane, M., "UniProt: The Universal Protein Knowledgebase in 2021", _Nucleic Acids Res._, *49*, D480–D489 (2020).
  
# License

## Apache License 2.0

A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code. The full copy of the license is available in [the repository](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/LICENSE/license.txt).
