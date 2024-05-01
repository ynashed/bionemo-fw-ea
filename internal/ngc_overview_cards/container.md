## What is BioNeMo?

BioNeMo Framework provides versatile functionalities for developing, training and deploying large language models. BioNeMo allows users to build biomolecular models by providing access to pre-trained models, creating workflows to fit downstream task models from embeddings, and generating biomolecules that meet user-specified criteria based on the fit model. Built for supercomputing scale, the framework allows developers to easily configure and deploy distributed multi-node jobs with minimal code.

This container contains implementation of the following models : 
* EquiDock
* DiffDock
* ESM-2nv 
* ESM-1nv
* ProtT5
* MegaMolBART

This container also supports downstream tasks like Protein secondary structure prediction, Protein thermostability using Meltome Atlas, Subcellular localization, Retrosynthesis and fine-tuning on FLIP /PhysChem datasets.


### Software Requirements

BioNeMo is only supported on x86 Linux systems. If you are not on such a system, you must use the project's Docker images to develop and execute BioNeMo code.

System Requirements:

* Docker (with GPU support, docker engine >= 19.03).
* Python 3.10 or above
* Pytorch 1.13.1 or above
* NeMo pinned to version 1.20
* NVIDIA GPU, if you intend to do model training. BioNeMo is compatible with most NVIDIA GPUs, with some incompatibility:
Tested GPUs:
    * DGX-H100, A100, V100
    * RTX A6000, A8000
    * Tesla T4
    * GeForce RTX 2080 Ti
* GPUs with known issues:
    * Tesla K80
* bfloat16 precision requires an Ampere generation GPU or higher.

### Getting Started
For information about how to get started with BioNeMo refer to the [documentation](https://docs.nvidia.com/bionemo-framework/latest)

### License

By pulling and using the container, you accept the terms and conditions of the [NVIDIA Software Evaluation License Agreement](https://developer.download.nvidia.com/licenses/NVIDIA-BioNeMo-Framework-Evaluation-Software%20License(14Nov2023).pdf).
