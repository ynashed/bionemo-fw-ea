# Prott5nv

# Model Overview

## Description:

Prott5nv provides numerical embeddings for each amino acid in the input protein sequence. Prott5nv is a protein language model whose output embeddings can be used for predictive downstream tasks. Prott5nv has a T5 encoder-decoder architecture. It was developed using the BioNeMo framework and trained starting from a model pre-trained on NLP data. This model is ready for commercial use.

<br>

## References:

[1] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." The Journal of Machine Learning Research 21.1 (2020): 5485-5551.

[2] Elnaggar, Ahmed, et al. "Prottrans: Toward understanding the language of life through self-supervised learning." IEEE transactions on pattern analysis and machine intelligence 44.10 (2021): 7112-7127.
<br> 

## Model Architecture: 
**Architecture Type:** Transformer,  Encoder-Decoder<br>
**Network Architecture:** T5 <br>

The model has 12 layers, 12 attention heads, a hidden space dimension of 768, and contains 192M parameters.  Pre-norm layer normalization and Gaussian Error Linear Unit (GELU) activation are used throughout.

## Input:
**Input Type(s):** Text (Protein sequences) <br>
**Input Format(s):** Comma Separated Values <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** Maximum input sequence length is 512 tokens. Proteins whose amino acid sequence is longer than this are truncated at 512 amino acids.  Protein sequences are tokenized with each token represented as an integer id. <br>

## Output:
**Output Type(s):** Text, Numerical <br>
**Output Format:** 2D <br>
**Other Properties Related to Output:** Maximum output length is 512 embeddings. Embeddings are represented as floating point 32 numeric vectors. <br> 

## Software Integration:
**Runtime Engine(s):** 
* BioNeMo, NeMo 1.2 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>

## Model Version(s): 
prott5nv.nemo, version 23.06 <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** [UniRef50](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50)  <br>

**Data Collection Method by dataset** <br>
* [Human] <br>

**Labeling Method by dataset** <br>
* [Not Applicable] <br>

**Properties:** 
UniRef50 (release 05/2022) was used for training. UniRef50 is one of the UniProt Reference Clusters (UniRef) clustered sets of sequences from the UniProt Knowledgebase (including isoforms) and selected UniParc records in order to obtain complete coverage of the sequence space at several resolutions. It was generated by clustering another UniRef clustered set called UniRef90 and selecting one reference sequence from each cluster. Sequences longer than the maximum sequence length of 512 were removed, resulting in approximately 46M protein sequences. The sequences were randomly split with 4.35K sequences in validation loss calculation during training and 875K sequences in testing and 45.1M sequences used exclusively in training. <br>


## Evaluation Dataset:
**Link:** [FLIP – secondary structure, conservation, subcellular localization, meltome, GB1 activity](http://data.bioembeddings.com/public/FLIP/fasta/)  <br>
**Data Collection Method by dataset** <br>
* [Human] <br>
* [Automatic/Sensors] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human & Automated] <br>

**Properties:** 
The FLIP datasets evaluate the performance of the model on five specific downstream tasks for proteins. It provides pre-defined splits for fine-tuning a pretrained model using task-specific train and validation examples, and subsequently evaluating it on a task-specific test split.

The secondary structure dataset contains experimental secondary structures of proteins with 9712 proteins for training, 1080 proteins for validation, and 648 proteins for testing. 

Conservation dataset contains conservation scores of the residues of protein sequences with 9392 proteins for training, 555 proteins for validation, and 519 proteins for testing. 

Subcellular localization dataset contains protein subcellular locations with 9503 proteins for training, 1678 proteins for validation, and 2768 proteins for testing. 

Meltome dataset contains experimental melting temperatures for proteins, with 22335 proteins for training, 2482 proteins for validation, and 3134 proteins for testing. 

Finally, GB1 activity dataset contains experimental binding affinities of GB1 protein variants with variation at four sites (V39, D40, G41 and V54) measured in a binding assay, with 6289 proteins for training, 699 proteins for validation, and 1745 proteins for testing. <br>

## Inference:
**Engine:** BioNeMo, NeMo <br>
**Test Hardware:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta] <br>

Using the T5 model trained on NLP data, the model was further pre-trained with protein sequences using data parallelism on 224 V100 GPUs for 58 epochs (approximately 970189 iterations) using a micro batch size of 12 molecules per GPU. The total training time was approximately 120 wall-clock hours. Inverse square root annealing was used, with a minimum learning rate of 0.0 and ~10000 warmup steps. Fused Adam optimization was used with parameters β1=0.9 β2=0.999 and weight decay=0.01. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [Insert Link to Model Card++ here].  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## License 
ProtT5nv is provided under the {{model_license_slug}}.