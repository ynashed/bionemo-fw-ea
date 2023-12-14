# DiffDock

## Model Overview

DiffDock is an equivarient geometric model capable of generating ligand poses on a protein surface - suitable for blind docking tasks. Rather than a regressive model that uses information to train to a specific site, DiffDock has the capability to generate full 3D-coordinates of a ligand given a protein structure to dock in through a score-based diffusion model. This is the functionality of the `Score Model`. In addition, DiffDock comes trained and prepared with a `Confidence Model` that is used to rank generated poses based on the likelihood they are below 2 Å RMSD. When combined in `Inference` they make for a powerful tool in candidate generation and pose ranking.

## Intended Use

![dd_overview](../images/diffdock_fw_overview.png)

The core of the DiffDock inputs are the structural representation of a molecule, and a protein. In its current state, both `.mol2`, and `.sdf` file formats for ligands are supported in training, while either structural or SMILES representations are supported for inference. File formats for protein structures are required to be in `.pdb` file format for training, but can be given in `FASTA` format for folding experiments. ESM embeddings as a featurizer are used from a collection of `.FASTA` files.  

The current DiffDock architecture within BioNeMo allows for the following:
1. Data preprocessing utilities for protein and ligand files into computational graphs, and generating ESM embeddings for protein featurization 
2. Training both Score and Confidence models 
3. Using pre-loaded or newly trained checkpoints for inference

DiffDock is a multi-component model, and therefore the setup process for full-fledged inference requireds some additional work beyond traditional models. Once your ground truth data has been assembled, the `train.py` script handles each step of data preprocessing, and model training. These options can be changed with the appropriate `do_embedding_preprocessing`, `do_preprocessing`, and `do_training` Flags. In addition, the use of configuration parameters are stored in `conf` directories with respective titles for `embedding_preprocess.yaml`, `train_[score/confidence].yaml`.  After successful training, it is helpful to gain an understanding of how your model will perform in true inference outside of training data. The first step is to evaluate the confidence of generated predictions against known data. Trained score model weights are used to generate poses through the reverse diffusion process. These generated poses can be evaluated, and stored for confidence model training. In pure inference, with trained score and confidence models, the score model will generate poses through reverse diffusion, and based on the evaluated generated dataset, a trained confidence model will rank the poses from 1-N. 

## Model Architecture

The DiffDock architecture makes uses of convolutional networks based on tensor products of irreducible representations (irreps) of SO(3) {cite:p}`thomas2018tensorfield` for both score and confidence models. The `e3nn` library from {cite:p}`geiger2022e3nn` at its core. 
Both the architectures can be decomposed into three main parts: embedding layer, interaction layers, and output layer. We outline each of them below.

1. The Embedding Layer<br>
Structures are represented as heterogeneous geometric graphs with nodes representing ligand (heavy) atoms, receptor residues (located in the position of the α-carbon atom), and receptor (heavy) atoms (only for the confidence model).

2. Interaction Layers<br>
At each layer, for every pair of nodes in the graph, we construct messages using tensor products of
the current node features with the spherical harmonic representations of the edge vector. 
This operation remains one of the most computationally intensive in the model. The BioNeMo FW targets this computation at the core through optimized CUDA kernel operations in both forward and backward of the tensor product. 

3. The Output Layer<br>
The output of the score model is in tangent space, and includes representations for the translational, rotational, and torsion scores of the ligand. The confidence model outputs a single SE(3)-invariant scalar representing the confidence score.

## Limitations
DiffDock is currently restricted to static snapshot understanding of single ligand and protein interactions. For more involved systems included multi-ligands in a single protein pocket, multiple protein pockets without a ligand blocker, DiffDock inference may perform poorly due to the unaware implications of ligand-ligand interactions in solvent. Because ESM2 is used as a featurizer, some non-standard amino acids are ignored in the process. 

Size-aware batch sampler is used because the memory from the cross-graph (ligand-receptor) fluctuate a lot.  Size-aware batch sampler estimate the upper bound of memory usage from estimating the upper bound of the number of cross graphs, and do sampling or prepare batch of samples to avoid Out-Of-Memory error.

Primary data limitations for the original implementation of DiffDock arise in non-uniform sizes from the cross-graph (ligand-receptor). In the forward diffusion, ligands are randomly perturbed, and a dynamic cutoff is used to build cross graph between ligand and receptor, this results in a dynamic cross graph that has fluctuating edges, depending on the diffusion time and resulting in fluctuating memory usage. This has been solved in the BioNeMo FW implementation `SizeAwareBatchSampler`. This setting can be used to estimate the upper bound of memory usage from estimating the upper bound of the number of cross graphs, and do sampling or prepare batchs of samples to avoid Out-Of-Memory error.

## Suggested Reading
Much found in the `conf/train*.yaml` files overlap with previous definitions of models in the BioNeMo framework (items like device requirements, data loading, etc.). Others pertain to well defined characteristics of the original publication of the DiffDock model like architecture design associated with nodes and edges of the molecular graph, diffusion model hyperparameters, etc. We suggest a thorough read of the authors original publication on arXiv [here](https://arxiv.org/pdf/2210.01776.pdf).

## License

DiffDock is provided under the {{model_license_slug}}.