# Introduction

AI models are changing how we think about and access information on an unprecedented scale. These methods, ranging from regression, classification, and even generation, allow the exploration of data-driven domains in unique ways. BioNeMo seeks to serve as a platform for accessibility to multiple bio-related AI tools to serve researchers in their challenges. The BioNeMo framework facilitates centralized model training, optimization, fine-tuning, and inferencing for protein and molecular design.

![](./images/bionemo_overview_2.png)

## BioNeMo Framework
BioNeMo Framework provides versatile functionalities for developing, training and deploying large scale bio-based models. BioNeMo allows users to build biomolecular models by providing access to pre-trained models, creating workflows to fit downstream task models from embeddings, and generating biomolecules that meet user-specified criteria based on the fit model. Built for supercomputing scale, the framework allows developers to easily configure and deploy distributed multi-node jobs with minimal code. 

![](./images/bionemo_overview_1.png)

The underpinnings of the biological research framework rely on [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html), a framework initially established for conversational AI methods. NeMo provides a robust environment for working with large learning models, including [Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) models. The BioNeMo Framework provides enhancements to PyTorch Lighting, such as hyperparameter configuarbility with YAML files and checkpoint management. Users can conveniently and quickly train models using these features, test them for desired tasks, and integrate them alongside existing applications. 

Some of the key features of BioNeMo Framework are:

* Development and training of large transformer models using NVIDIA's Megatron framework.
* Easy to configure multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision.
* Model training recipes that can be readily implemented on DGX compute infrastructure.
* Logging with Tensorboard and Weights and Biases to monitor the model training process.

## Prerequisites
Please follow the instructions on the [pre-requisites documentation](./pre-reqs.md) page.

## BioNeMo Models: Overview


| **Model**                                               | **Modality**             | **Uses**                                      |**Trained/Converted Checkpoints on NGC**           |
|---------------------------------------------------------|--------------------------|-----------------------------------------------|---------------------------------------------------|
| [DiffDock Score Model](./models/diffdock.md)            | Protein + Molecule       | Generation of Ligand Poses                    | [Public checkpoint converted to BioNeMo format](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/diffdock_score)     |
| [DiffDock Confidence Model](./models/diffdock.md)       | Protein + Molecule       | Generation of Ligand Poses                    | [Public checkpoint converted to BioNeMo format](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/diffdock_confidence)        |
| [EquiDock DIPS Model](./models/equidock.md)             | Protein                  | Protein-Protein Complex Formation             | [BioNeMo checkpoints pre-trained from scratch](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/equidock_dips)               |
| [EquiDock DB5 Model](./models/equidock.md)              | Protein                  | Protein-Protein Complex Formation             | [BioNeMo checkpoints pre-trained from scratch](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/equidock_db5)                |
| [ESM-2nv 650M](./models/esm2-nv.md)                     | Protein                  | Representation Learning                       | [Public checkpoint converted to BioNeMo Format](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv650m)                  |
| [ESM-2nv 3B](./models/esm2-nv.md)                       | Protein                  | Representation Learning                       | [Public checkpoint converted to BioNeMo Format](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv3b)                    |
| [ESM-1nv](./models/esm1-nv.md)                          | Protein                  | Representation Learning                       | [BioNeMo checkpoints pre-trained from scratch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm1nv)                       |
| [ProtT5nv](./models/prott5nv.md)                        | Protein                  | Representation Learning                       | [BioNeMo checkpoints pre-trained from scratch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/prott5nv)                     |
| [MegaMolBART](./models/megamolbart.md)                  | Small Molecule           | Representation Learning + Molecule Generation | [BioNeMo checkpoints pre-trained from scratch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/megamolbart)                  |

For more information about the models included in BioNeMo Framework, refer to the Model Cards linked in the table above or the original publications referenced in the respective model descriptions.

Refer to the [Quickstart Guide](./quickstart-fw.md) for details on how to get started with BioNeMo Framework.
