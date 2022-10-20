# What is BioNeMo?

This container performs pre-training and inference on BioNeMo models.

## Software Requirements

The BioNeMo container is assembled with the prerequisite software for pre-training and performing inference using one of the BioNeMo models.

Requirements:

- Linux OS
- Pascal, Volta, Turing, or an NVIDIA Ampere architecture-based GPU.
- NVIDIA Driver
- Docker

## Getting Started

This container encloses all prerequisites for training and inference. Please follow these steps to start the container for inference.

Start an instance of the Docker image using the following command:

```
# For Docker version 19.03 or later
docker run \
   --gpus all \
   --rm \
   -p 8888:8888 \
   nvcr.io/t6a4nuz8vrsr/bionemo:latest


# For Docker version 19.02 or older
docker run \
   --runtime nvidia \
   --rm \
   -p 8888:8888 \
   nvcr.io/t6a4nuz8vrsr/bionemo:latest
```

In a browser open URL http://<<HOSTNAME/IP>>:8888

Additional information, including how to run training, is available here:

- [README](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/README.md)
- [Quickstart](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/QUICKSTART.md)
- [Source Code](https://gitlab-partners.nvidia.com/clara-discovery/bionemo)

## License
By pulling and using the container, you accept the terms and conditions of the [Apache 2.0 license](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/LICENSE/license.txt).
