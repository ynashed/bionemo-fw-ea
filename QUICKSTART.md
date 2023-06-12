# Quickstart Guide

## Introduction

The following quickstart guide contains configuration information and examples of how to run data processing and training of BioNeMo models on a workstation. The [README](./README.md) contains additional information that will be of use for more advanced tasks, such as model development or changes to the configuration. There is also an example script for [SLURM jobs](./examples/molecule/megamolbart/scripts/slurm/megamolbart_pretrain_slurm.sh) available. 

## Logging with Tensorboard and Weights and Biases

NeMo provides optional logging with Tensorboard and Weights and Biases. Use of Weights and Biases is optional, but highly recommended. All that is required is an account and an [API key](https://docs.wandb.ai/guides/track/public-api-guide). The API key should be setup on the workstation or computer from which the job is launched as described on the [Weights and Biases website](https://docs.wandb.ai/guides/track/public-api-guide#authentication).

## Configure `launch.sh` Script

The [`launch.sh` script](./launch.sh) can be used for a number of tasks, including pulling and running the BioNeMo container. First, clone this repo to your workstation to create the project directory. The script requires a settings file called `.env`. This file will automatically be created in the project directory if it does not exist on first launch, but below is an example of the file. If created manually, it should be named `.env` and placed inside the project directory. All of the variables are described in the `usage` section of [`launch.sh`](./launch.sh) in this directory. Missing variables will be substituted for the defaults in the script.

```bash
BIONEMO_IMAGE=nvcr.io/nvidian/clara-lifesciences/bionemo_ci:latest       # Container with tag
BIONEMO_PATH=/workspace/bionemo                                          # Location of code to be used /workspace/bionemo or /opt/nvidia/bionemo
PROJECT_PATH=$(pwd)                                                      # Path of env config and optional development code
DATA_PATH=${HOME}/data                                                   # Local path to save downloaded and processed data
RESULT_PATH=${HOME}/result/bionemo_experiments                           # Path for training results
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)              # WandB logging requires API key configuration
JUPYTER_PORT=8888                                                        # Jupyter port for inference notebooks
PROJECT_MOUNT=/workspace/bionemo                                         # Location of library in container /workspace/bionemo for dev work or /opt/nvidia/bionemo for non-dev use                                      
GITHUB_ACCESS_TOKEN=INSERT_GITHUB_ACCESS_TOKEN_HERE                      # Only required for building container from a private branch
GITHUB_BRANCH=v0.3.0_ea1                                                 # GitLab branch
REGISTRY=nvcr.io                                                         # Only required for registry login
REGISTRY_USER='$oauthtoken'                                              # Only required for registry login
NGC_CLI_API_KEY=$(grep -m 1 apikey ~/.ngc/config | head -n 1 | cut -d' ' -f3) # Requires NGC cli configuration
NGC_CLI_ORG=nvidian
NGC_CLI_TEAM=clara-lifesciences
NGC_CLI_FORMAT_TYPE=ascii
```


## Docker Container

The latest docker container should be downloaded from NGC. If the `launch.sh` script is configured, it can be downloaded with `bash launch.sh pull`.

Once a container has been pulled and the `.env` file has been setup, it can be started in interactive mode with `bash launch.sh dev`. Data processing and model training must be performed inside the Docker container.

## Data Processing and Training

In order to prevent downloading and processing of data everytime a new container is launched, be sure to set a valid path at DATA_PATH variable in the .env file. The data will be mounted at `/data` location inside the container and be available for use. The following files are required for data processing or training models:

### Model Configuration File

BioNeMo uses YAML based parameter files. The existing model configuration files can be found in the corresponding `conf` subdirectory of `examples`, e.g. `examples/molecule/megamolbart/conf`. Additional files can be created to suit other configurations. Though data processing does not require an explicit model configuration, one of these files (or the default) must be provided.

### Python Training Script

A Python-based pretraining script reads the YAML parameter files. These scripts are found in the corresponding subdirectory of `examples`, e.g. `examples/molecule/megamolbart/scripts/pretrain.py`. These scripts run the data processing and training. Typically changes will not need to be made to this file.

### Job Execution

The name of the YAML file and any additional configuration parameters can be provided on the command line. By changing these values, it is possible to configure the size of the training run (number of nodes and gpus), model precision, etc. Note that multi-node training has been validated only on SLURM-based systems.

Some commonly changed configuration parameters are:

```yaml
trainer:
  devices: 8 # number of GPUs per node
  num_nodes: 8 # number of nodes

model:
  micro_batch_size: 8 # batch size for each model instance
```

For convenience, these commands have been wrapped into a series of shell scripts that demonstrate functionality by model and are located in the corresponding `scripts` directory within `examples`:

- MegaMolBART: `examples/molecule/megamolbart/scripts/pretrain_quick.sh`
- ESM-1nv: `examples/protein/esm1nv/scripts/pretrain_quick.sh`
- ProtT5nv: `examples/protein/prott5nv/scripts/pretrain_quick.sh`

These scripts first require some variables to be set in their `CONFIG` section. They also take optional flags and a command. The flags are:

- `-c`, `--config`: Path to a YAML config file. The default is specified in each of the respective scripts.
- `-f`, `--data-files`: The set or subset of processed data files (in CSV format) to use for training. Consecutive ranges of files can be supplied -- for three files (x000.csv, x001.csv, and x002.csv), the range can be supplied as "x[000..002]". However, the addition of the required shell escapes to this can become complicated, so it's adviseable to use `_OP_` instead of `[` and `_CL_` instead of `]`, making the range `x_OP_000..002_CL_`
- `-o`, `--result_dir`: Set the path for the results.
- `-a`, `--args`: Additional arguments passed to the training. Use `--args` flag followed by a string of additional arguments, e.g. to use tensor model parallelism of 2 with 4 GPUs: `--args "++trainer.devices=4 ++model.tensor_model_parallel_size=2"`

The two command options are `preprocess`, which will download and process the data, and `train` which performs pre-training using the data. An example is provided for executing one of the scripts once the `CONFIG` section has been edited:

```bash
cd examples/protein/esm1nv/scripts && \
bash pretrain_quick.sh \
--config esm-1nv_config \
--data-files x_OP_000..010_CL_ \
train
```

For preprocessing MegaMolBART data, a large amount (~100GB) of data in the form of multiple files must be downloaded for the selected ZINC15 tranches. To test the preprocessing and training on a subset of the files, change the filename in the `links_file` setting in the [config file](./examples/molecule/megamolbart/conf/pretrain_base.yaml) from `ZINC-downloader.txt` to `ZINC-downloader-sample.txt`, e.g.:

```yaml
links_file: '${oc.env:PROJECT_MOUNT}/examples/molecule/megamolbart/dataset/ZINC-downloader-sample.txt' # to process a subset of the ZINC15 data
```

For more advanced customization, additional training arguments can be added to the shell script. Alternatively, the Python training commands executed by the script can be run directly.

## Training Output

### Log Files

NeMo creates a number of log files during training in the results directory:

- `cmd-args.log` : the Python commands and any additional parameter overrides used for training
- `hparams.yam` : the final set of all hyperparmeters used for training
- `git-info.log` : the commit hash of the code used for training and the results of `git diff` on this code to store any additional modifications from this commit
- `nemo_log_globalrank-0_localrank-0.txt` : NeMo creates a number of logs, depending on how many GPUs are used in training. Local rank is the rank order within a node and global rank is the rank order across all nodes.
- `nemo_error_log.txt` : error logs from NeMo
- `lightning_logs.txt` : PyTorch Lightning logs

### Checkpoints

Checkpoints are stored in the `checkpoints` directory and are managed by NeMo. For example, it is possible to ensure only the top k checkpoints are saved. See the NeMo documentation for more information. NeMo can also optional create a `*.nemo` checkpoint which has the optimizer states removed and can be used for inference or fine tuning tasks. Checkpoints are automatically reloaded when resuming training.

### Tensorboard and Weights and Biases

NeMo can create a Tensorboard file in the results directory, if logging to Tensorboard has been enabled. The Weights and Biases loggs will be created in a directory called `wandb`, if Weights and Biases logging is enabled, and can optionally be uploaded after training or from another computer if cluster restrictions do not allow it to be done during training.

## Inference 

Trained BioNeMo models are provided on NGC for use. These models can be loaded with a gRPC interface provided in BioNeMo. Example notebooks are provided in the corresponding `nbs` directory in `examples`, e.g. `examples/molecule/megamolbart/nbs`. First, download the pretrained models with `bash launch.sh download`, which will require an [NGC API key](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key) to be configured. Alternatively, the models can be downloaded inside the container by running `source download_models.sh && download_bionemo_models`. Then, to launch the gRPC interface and a Jupyter notebook service in the corresponding directory, ensure that `JUPYTER_PORT` is set correctly in the `.env` file, then run `bash.launch.sh dev -c <model_name>`, where `<model_name>` is one of the following: `megamolbart`, `prott5nv`, `esm-1nv`. Then open a browser at `http://<<HOST_NAME>>:$JUPYTER_PORT` to execute the notebook.
