# Hardware and Software Prerequisites

Before you begin using the BioNeMo framework, ensure the following prerequisites are met.

## Hardware

The BioNeMo Framework is compatible with environments that have access to NVIDIA GPUs. Bfloat16 precision requires an Ampere generation GPU or higher. Tested GPUs include: H100, A100, and RTX A6000. There is mixed support for GPUs without bfloat16 support, such as V100, T4, Quadro RTX 8000, and GeForce RTX 2080 Ti. GPUs with known issues include: Tesla K80.

## Software

The BioNeMo Framework is supported on x86 Linux systems. Please use the project’s Docker images to develop and execute the code.

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
