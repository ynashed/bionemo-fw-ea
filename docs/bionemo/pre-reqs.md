# Prerequisites for Using BioNeMo Framework

Before you begin using the BioNeMo framework, ensure the following prerequisites are met. 

BioNeMo is only supported on x86 Linux systems. If you are not on such a system, you must use the project's Docker images to develop and execute BioNeMo code.
System Requirements:

Docker (with GPU support, docker engine >= 19.03).
Python 3.10 or above
Pytorch 1.13.1 or above
NeMo pinned to version 1.20
NVIDIA GPU, if you intend to do model training. BioNeMo is compatible with most NVIDIA GPUs, with some incompatibility:

Tested GPUs:
DGX-H100, A100, V100
RTX A6000, A8000
Tesla T4
GeForce RTX 2080 Ti

GPUs with known issues:
Tesla K80

bfloat16 precision requires an Ampere generation GPU or higher.

To request access : 

1. Access to [BioNeMo Framework](https://www.nvidia.com/en-us/gpu-cloud/bionemo/) from NVIDIA.
2. Access to NVIDIA DGX compute infrastructure ([DGX-Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/) or [DGX-Pod](https://www.nvidia.com/en-us/data-center/dgx-basepod/)).

## How to Run the Example Commands?
All BioNeMo Framework (`bionemo`) code **must** be run within containers created from the official release images.
Some documented examples will explicitly call out whether a command is expected to be executed from the container or not.
However, if this isn't specified, as a general rule when reading the documentation, whenever one sees commands of the form `python <a .py file defined in bionemo>`, the expected execution environment is the container.

However, not every documented command requires the container.
Again, sometimes we will explicitly label the execution environment for the command.
However, if this isn't specified, as a general rule when reading the documentation, whenever one sees a command that _doesn't_ start with `python` -- such as `ngc`, `bcprun`, etc. -- it is ok to assume that one should run the command from _outside_ the container.
As in, on one's host machine.
