# Quickstart Guide

The purpose of this quickstart is to make users familiar with the different components, features and functionalities of the BioNeMo framework. It outlines how to access various resources related to BioNeMo, what is provided inside the BioNeMo container, and how a user can adapt different parts of BioNeMo for their intended use case.


## Setup
### Accessing BioNeMo Framework

To get the access to BioNeMo Framework, visit the {{bionemo_info_url}}.
Request access to NVIDIA DGX compute infrastructure ([DGX-Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/) or [DGX-Pod](https://www.nvidia.com/en-us/data-center/dgx-basepod/)) to run full scale training / inference jobs.

### NGC Setup

BioNeMo Framework assets as well as compute needed to run the model training/inference are available via [NVIDIA NGC](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html)

1. Sign into [NGC](https://catalog.ngc.nvidia.com/).
2. Download and install [NGC CLI](https://ngc.nvidia.com/setup) and setup your [API Key](https://ngc.nvidia.com/setup/api-key).

### [Optional] Weights and Biases Setup

Training progress and charts of the models can be visualized through [Weights and Biases](https://docs.wandb.ai/guides/track/public-api-guide). Setup your [API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) to enable logging.

## Getting Started on Local Machine
Use the guide below to explore the contents of the BioNeMo container and may be run small experiments. Any full scale training/inference jobs should be run on the cloud for better performance and scalability.

### Pull the Docker Container
This step assumes you have Docker installed on your workstation.

```bash
docker login {deploy_ngc_registry}
Username: $oauthtoken
Password <insert NGC API token here>
```
Details related to the BioNeMo container addreess on NGC, organization ID, Team, (and ACE in case of DGX-Cloud) will be provided to the users in NGC invitation e-mail. To pull the container, execute the following, where  `{deploy_ngc_org}` and `{deploy_ngc_team}` should be updated with the correct NGC org and team names, respectively:

```bash
docker pull {deploy_ngc_registry}/{deploy_ngc_org_team}/{deploy_container_name}:{deploy_container_tag}
```

### Run a Sample Training Job

For the purposes of this example, we use MegaMolBART as our model.

#### Step 1: Lauch the Docker Container

```bash
docker run -it --rm --gpus all {deploy_ngc_registry}/{deploy_ngc_org_team}/{deploy_container_name}:{deploy_container_tag} bash
```

#### Step 2: Pre-process the data

```bash
cd /workspace/bionemo/examples/molecule/megamolbart && python pretrain.py --config-path=conf --config-name=pretrain_xsmall_span_aug do_training=False model.data.links_file='${oc.env:BIONEMO_HOME}/examples/molecule/megamolbart/dataset/ZINC-downloader-sample.txt' model.data.dataset_path=zinc_csv
```
BioNeMo currently offers offline data pre-processing to enable faster loading of data samples during training/inference. Raw data is pre-processed, memory mapped and stored on disk. This pre-processed data is then ingested into the training/inference pipeline. This also allows for exact resuming of data samples when resuming from a checkpoint etc. There is no online data pre-processing at the moment (meaning, data processing cannot be done during training).

#### Step 3: Train

```bash
python pretrain.py --config-path=conf --config-name=pretrain_xsmall_span_aug do_training=True model.data.dataset_path=zinc_csv model.data.dataset.train=x000 model.data.dataset.val=x000 model.data.dataset.test=x000 exp_manager.exp_dir=results
```
BioNeMo creates a number of log files during training in the results directory:

- `cmd-args.log` : the Python commands and any additional parameter overrides used for training
- `hparams.yaml` : the final set of all hyperparameters used for training
- `git-info.log` : the commit hash of the code used for training and the results of `git diff` on this code to store any additional modifications from this commit
- `nemo_log_globalrank-0_localrank-0.txt` : BioNeMo creates a number of logs, depending on how many GPUs are used in training. Local rank is the rank order within a node and global rank is the rank order across all nodes.
- `nemo_error_log.txt` : error logs from NeMo
- `lightning_logs.txt` : PyTorch Lightning logs

Checkpoints are stored in the `checkpoints` directory. The latest checkpoint is automatically reloaded from the `checkpoints` directory when resuming training. A `*.nemo` checkpoint could also be saved which has the optimizer states removed and can be used for inference or fine-tuning tasks.

BioNeMo supports optional logging with Tensorboard and Weights and Biases. Tensorboard logging will create the standard `events.out.tfevents.....` file, while Weights and Biases logs are stored in a directory called `wandb`. Weights and Biases logging can be done offline if cluster restrictions do not allow upload of files during training. To enable offline logging and upload the contents after training, follow the directions for [offline sync](https://docs.wandb.ai/guides/technical-faq/setup#can-i-run-wandb-offline).

## Large Scale Training / Inference
Inorder to run training / inference jobs at scale, checkout [Getting Started with BCP](./quickstart-bcp.md)
