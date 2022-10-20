# README

## Introduction

NVIDIA BioNeMo is a framework for training and deploying large biomolecular language models at supercomputing scale for the discovery and development of theraputics. The large language model (LLM) framework currently has models for small molecules (SMILES) and protein sequences. More information about the models and their training is available in the [model guides](./docs/ngc/models).

BioNeMo relies on [NeMo](https://github.com/NVIDIA/NeMo). NeMo provides a robust environment for developing, training, and deploying deep learning models, including Megatron models. NeMo provides enhancements to PyTorch Lighting such as hyperparameter configuarbility with YAML files and checkpoint management. It also enables the development and training of large transformer models using NVIDIA's [Megatron](https://github.com/NVIDIA/Megatron-LM) framework, which makes multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision easily configurable. The [NeMo User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) contains more information about all of these features. It is highly suggested to review at least the [NeMo Fundamentals](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) tutorial to understand how to configure jobs.

## Repo Organization

The BioNeMo repo is organized by biological entity (molecule, protein) and by deep learning model, e.g. `molecule/megamolbart` or `protein/esm1nv`. Shared code can be found within `utils` files or directories. These are the most important directories and their purpose:

- `bionemo`: The library location, including functionality for data processing, data loading, models, and associated functions.
- `examples`: The files required to run a training or inferences: Python training scripts, YAML configuration files, shell scripts, and Jupyter notebooks.
- `models`: The location for tokenizers and trained models that are downloaded.
- `setup`: Dockerfile and associated setup scripts.

## Quickstart

The [Quickstart Guide](./QUICKSTART.md) contains configuration information and examples of how to run data processing and training of a small model on a workstation. The tutorial contained in the Quickstart is a good way to gain familiarity with how trainings are run and configured. The remainder of this README contains information that will be of use for more advanced tasks, such as code development or model configuration changes. Familiarity with the material in the [Quickstart Guide](./QUICKSTART.md) is assumed for the remainder of this guide.

## Code Development

`BIONEMO_PATH` is an environment variable used to select the path to the BioNeMo library in use. By default, this is the library installation path (`/opt/nvidia/bionemo`). For development, code should be mounted inside the container at `/workspace/bionemo`.

The easiest way to develop or test the training of in-development code is to mount a local copy of the code inside the docker container. The `launch.sh` script mounts the current working directory inside the container by default. This behavior can be customized by editing `PROJECT_PATH`, which is the local copy of the code, and `BIONEMO_PATH` in the `.env` file.

It may also be necessary to recompile the Megatron helpers, which can be done by running the script `setup/recompile_megatron_helper.sh`. This recompilation should also be done immediately before training starts on clusters as a best practice.

### Build and Start Container

The `launch.sh` script (see the [Quickstart Guide](./QUICKSTART.md#configure-launchsh-script) for configuration of the required `.env` file) can be used to build and push containers to a registry and start a container for interactive work. See the instructions inside the script for more information. 

Once the `.env` script is created, a container can be pulled by running `bash launch.sh pull`. The setup for BioNeMo training and inference is quite involved and requires the use of a Docker container. Due to this complexity, it is highly suggested to pull a prebuilt container. However, if a container must be built, it can be done by running `bash launch.sh build -p`. Executing this command will require an [NGC API key](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key) to be configured. If pushing to a registry is desired, `bash launch.sh push` will complete this task.

Once a container has been built, it can be started in interactive mode with `bash launch.sh dev`.

### Setup Data Processing and Training Files

See the [Data Processing and Training](./QUICKSTART.md#data-processing-and-training) section in the [Quickstart Guide](./QUICKSTART.md) for information on the files required for processing data and running a training. These files can be customized as required for different model sizes, data selection, and training configuration.

### Data Preprocessing

Data processing parameters can be customized in the YAML model configuration file or specified from the command line. The main change that is required to run data processing instead of training is to set `do_training` to `False`. This can be done from the command line by adding `++do_training=False` or by settting the following in the model's YAML config file:

```yaml
do_training: False # set to false if data preprocessing steps must be completed
```

Data processing only needs to be performed once for a given model. Additional configuration parameters are specific to each dataset / model and select changes are highlighted below for clarity. 

### Pre-Training

Once data processing is complete, training is performed by inverting the value of `do_training` to `True`. One additional feature of note is that ranges of data files can be selected. For example, to select a contiguous range of 100 data files, from x000.csv to x099.csv,, use the range indicator `x[000..099]`. For only ten files, use `x[000..009]`. Ensure these are set as appropriate for the train, validation, and test splits as below in the YAML config file:

```yaml
model:
  data:
    dataset:
      train: x[000..099]
      test: x[000..099]
      val: x[000..099]
```

**NOTE**: Do NOT add an extension to the data files. The appropriate extension is added automatically.

To change the datafiles selection from the command line, it's often easier to use a varation that does not involve brackets (`[`, and `]`) to avoid having to add multiple escape characters. For the range listed above, this would be `x_OP_000..099_CL_`.
