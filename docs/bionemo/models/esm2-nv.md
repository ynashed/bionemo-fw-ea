# ESM-2nv

## Model Overview

ESM-2nv is a model that has been trained on protein sequences. The embeddings from its encoder can be used as features for predictive models.

## Intended Use

Compute embeddings from input protein sequences. Embeddings are created for each amino acid in the protein sequence. Embeddings can then be used for downstream tasks such as prediction of secondary structure, subcellular localization, or others, as detailed by the FLIP benchmark tasks {cite:p}`dallago2021flip`.

## Model Architecture

ESM-2nv was developed using the BioNeMo framework by converting the ESM-2 model checkpoints from HuggingFace into the NeMo framework {cite:p}`lin2023evolutionary`. The underlying ESM-2 model uses an architecture called Bidirectional Encoder Representations from Transformers (BERT) and improves upon the ESM-1b model {cite:p}`lin2023evolutionary,rives2021esm` with various features including architectural changes, differences in embeddings, and custom transformations describe below.

ESM-2nv models are in principle compatible with ESM-2 checkpoints, meaning that ESM-2 public checkpoints from HuggingFace can be loaded into ESM-2nv architectures of similar size. We are making the 650M and 3B converted models available out-of-the-box in the current release. The 650M model has 33 layers, 20 attention heads, a hidden space dimension of 1280, and contains 650M parameters. The 3B model has 36 layers, 40 attention heads, a hidden space dimension of 2560, and contains 3B parameters.

Main improvements of ESM-2 over ESM-1b:
- **Positional Embeddings:** ESM-2 replaces the learnable absolute positional embeddings with rotary positional encoding (ROPE). This change enables support for sequences longer than the maximum length used during pre-training.

- **Masked Token Dropout:** Similar to how dropout layers scale the output during training, ESM-2 introduces the token-dropout scaling option. The motivation is to account for a model correction during evaluation when tokens are not masked. Technically, the model replaces the embeddings of [MASK] tokens with a vector of `0s`, and scales the non-masked embeddings using the following formula:
  $$e_i = \frac{1 - mask_{train}}{1 - mask_{observed}} \cdot e_{i},$$
  where:
  - $e_i$: token embeddings at position $i$,
  - $mask_{train}$ is a constant equal to 0.15 * 0.80, and
  - $mask_{observed}$ is the ratio of masked tokens in the input sequence with length $L$, denoted as $\frac{N_{[MASK]}}{L}$.

- **Attention-Masked Embeddings:** ESM-2 applies attention masking to the input embeddings to ignore padded positions.

- **Attention Query Scaling:** Instead of scaling the attention logits by $\sqrt{head_{size}}$ as in the original BERT, ESM-2 scales the query vector $Q$ by the same factor.

- **Custom GELU Function for the MLP Layer:** ESM-2 implements a slightly different function compared to F.gelu. It is defined as:
  $$x \rightarrow x \cdot 0.5 \cdot \text{torch.erf}\left(\frac{x}{\sqrt{2}}\right).$$

## Improvements in ESM-2nv over ESM-2

ESM-2nv models are HuggingFace ESM-2 model checkpoints that have been converted into NeMo-optimized BioNeMo model checkpoints.
ESM-2nv achieves the same performance benchmarks as ESM-2 but is optimized to provide faster training and inference on NVIDIA GPUs.
ESM-2nv enables customization for pre-training and inference parameters through YAML configuration files at time of model instantiation.
A complete, curated pre-training dataset is provided with the BioNeMo framework release of ESM-2nv to facilitate pre-training from scratch.

## Differences in ESM-2nv compared to ESM-2

Unlike ESM-2 pre-training data, the curated pre-training dataset provided with ESM-2nv release contains hits for *de novo* proteins, since sequences in UniRef100, UniRef90, and UniRef50 with high sequence similarity to a non-public 81 *de novo* proteins {cite:p}`lin2023evolutionary` are not filtered.

## Training

### Dataset and Processing

UniRef50 release 04/20221 was used for training {cite:p}`uniprot2021`. The representative sequence for each cluster was selected, resulting in approximately 49M protein sequences. The sequences were randomly split with 250K sequences in validation and the remaining in train. All train sequences that matched a validation sequence with 50% sequence identity were removed from the train set, resulting in 49,425,807 train sequences. A sampling dataset of UniRef90 sequences was created based on any UniRef90 representatives and cluster members that had complete sequences available from UniRef90 or UniRef100, and filtered to UniRef90 sequences for clusters that corresponded to the UniRef50 train set. This UniRef90 dataset was combined with the filtered UniRef50 training dataset to create the sampling fasta file. A mapping file was created to enable rapid replacement of UniRef50 sequences with a sequence sampled uniformly from the corresponding records in the sampling fasta file during each training update. The UniRef50 training fasta was sorted in the order of occurrence of records in column 1 of the mapping file. The UniRef90+UniRef50 sampling fasta file was sorted in the order of occurrence of records in column 2 of the mapping file.
Protein sequences longer than 1024 amino acids were cropped to 1023 from sequence start {cite:p}`devlin2018bert`.

### Infrastructure and Configuration

ESM-2nv can be trained from scratch using the provided dataset and code. The ESM-2nv 650M and 3B checkpoints in the current release have been converted from the models provided by Lin, et. al {cite:p}`lin2023evolutionary` and made available at HuggingFace ([650M model](https://huggingface.co/facebook/esm2_t33_650M_UR50D), [3B model](https://huggingface.co/facebook/esm2_t36_3B_UR50D)).

## Benchmarks

Two ESM-2nv pretrained checkpoints are available for download -- the 650 million and 3 billion parameter models -- which are converted to the NeMo format from the publically available HuggingFace checkpoints. This section provides accuracy benchmarks for these models, as well as expected training speed performance. At this time models which were trained from randomly initialized weights within the BioNeMo framework are not provided, and producing these models is ongoing work. It has been verified that further training (both the full model and with frozen-weight finetuning) the ESM2-nv models starting with the provided checkpoints results in continues decrease in loss.

### Accuracy Benchmarks

The accuracy of ESM2-nv was measured using the Fitness Landscape Inference for Proteins (FLIP) benchmarks {cite:p}`dallago2021flip`, which involve training a downstream task on top of the ESM2-nv model and measuring some characteristic:

| FLIP type                       | Dataset           | Dataset Split for Measurement | Metric              | 650m  | 3b   |
|---------------------------------|-------------------|-------------------------------|---------------------|-------|------|
| Secondary Structure             | sequences.fasta   | test                          | accuracy (%)        | 85.2  | 85.7 |
| GB1                             | two_vs_rest.fasta | test                          | rmse (score)        | 1.29  | 1.28 |
| Residue Conservation            | sequences.fasta   | test                          | accuracy (%)        | 33.2  | 34.5 |
| Protein Melting Point (Meltome) | mixed_split.fasta | test                          | rmse (&deg;C)       | 7.19  | 6.68 |


### Training Performance Benchmarks

Training speed was tested on DGX-A100 and DGX-H100 systems, on GPUs with 80GB of memory.

![ESM2 benchmarks](../../readme-images/esm2_days_to_train.png)

## License

ESM-2nv is as provided under the {{model_license_slug}}.
