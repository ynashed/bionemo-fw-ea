# Model Overview

ProtT5nv is a model that has been trained on protein sequences. The embeddings from its encoder can be used as features for predictive models, and sequence translation and generation tasks can utilize the full encoder-decoder architecture. 

## Model Architecture

The model uses an architecture called T5 and is based on the original ProtT5 model [1, 2]. Pre-norm layer normalization and GeLU activation are used throughout. The model has 12 layers, 12 attention heads, and a hidden space dimension of 768. Dropout was set to 0.1 during training.

## Training

ProtT5nv was developed using the NeMo framework starting from a model pretrained on NLP data. It was then further pre-trained with protein sequences using data parallelism on 224 V100 GPUs for 58 epochs (approximately 970189 iterations) using a micro batch size of 12 molecules per GPU. Inverse square root annealing was used, with a minimum learning rate of 0.0 and ~10000 warmup steps. Fused Adam optimization was used with parameters beta1=0.9 beta2=0.999 and weight decay=0.01. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

Sample training parameters can be found [here](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/examples/protein/prott5nv/conf/pretrain_small.yaml).

## Dataset

UniRef50 (release 05/2022) was used for training [3]. The reference sequence for each cluster was selected, with sequences longer than 512 amino acids removed, resulting in approximately 46M protein sequences. The sequences were randomly split with 4.35K sequences in validation, 875K sequences in test, and the remaining in train. Data masking was performed as described previously [2].

## How to Use this Model

The [Quickstart guide](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/QUICKSTART.md) has instructions for pre-training, and an example of how to use a pretrained model for inference can be found in [this notebook](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/examples/protein/prott5nv/nbs/Inference.ipynb).

## References

1. Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D., Rost, B., "ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing"' _IEEE T. Pattern Anal._, *1–1*, (2021).
2. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Materna, M., Zhou, Y., Li, W., Liu, P. J., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", _ArXiv_, (2019).
3. The Uniprot Consortium, Bateman, A., Marin, M. J., Orchard, S., Magrane, M., "UniProt: The Universal Protein Knowledgebase in 2021", _Nucleic Acids Res._, *49*, D480–D489 (2020).
  

# License

## Apache License 2.0

A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code. The full copy of the license is available in [the repository](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/LICENSE/license.txt).
