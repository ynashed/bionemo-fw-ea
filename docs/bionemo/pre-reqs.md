# Prerequisites for Using BioNeMo Framework

Before you begin using the BioNeMo framework, ensure the following prerequisites are met:

1. Access to [BioNeMo Framework](https://www.nvidia.com/en-us/gpu-cloud/bionemo/) from NVIDIA.
2. Access to NVIDIA DGX compute infrastructure ([DGX-Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/) or [DGX-Pod](https://www.nvidia.com/en-us/data-center/dgx-basepod/)).

Additionally, if you're using BioNeMo locally in a container, ensure that you're using the following:
* Linux OS
* Pascal, Volta, Turing, or an NVIDIA Ampere architecture-based GPU
* NVIDIA Driver
* CUDA toolkit installed
* Docker with Nvidia-DOcker support.

## How to Run the Example Commands?
All BioNeMo Framework (`bionemo`) code **must** be run within containers created from the official release images.
Some documented examples will explicitly call out whether a command is expected to be executed from the container or not.
However, if this isn't specified, as a general rule when reading the documentation, whenever one sees commands of the form `python <a .py file defined in bionemo>`, the expected execution environment is the container.

However, not every documented command requires the container.
Again, sometimes we will explicitly label the execution environment for the command.
However, if this isn't specified, as a general rule when reading the documentation, whenever one sees a command that _doesn't_ start with `python` -- such as `ngc`, `bcprun`, etc. -- it is ok to assume that one should run the command from _outside_ the container.
As in, on one's host machine.
