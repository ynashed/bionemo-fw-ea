
# Preparing Workspace and Data for Pre-training

This section outlines the steps to prepare your workspace with pre-processed files for training DiffDock using the NVIDIA NGC platform. 

This workspace can then be used to launch ESM2nv training job using the template script `<BioNeMO_Workspace>/examples/molecule/diffdock/scripts/train_bcp.sh`. 

For more information about how to launch DiffDock training using BCP script, you can check [Running BioNeMo on DGX-Cloud using BCP](./bcp-specific-commands-fw.md) tutorial.

The commands below expect that you have NGC CLI installed on your machine. All of the commands below can then be run outside of BioNeMo Docker container.

### Step 1: Create a Workspace
Create a new NGC workspace by running the following command. Make sure to replace `diffdock_bcp_train` with your desired workspace name and `nv-us-east-2` with your preferred ACE
```bash
ngc workspace create --name diffdock_bcp_train --ace nv-us-east-2
```

Store the workspace ID provided after creating the workspace. You will need this ID for mounting data and running jobs.

```bash
export WKSP_ID=??
```

### Step 2: Create a Local Data Directory
Create a local directory to store pre-training. For this documentation, we'll use `~/diffdock_data`.
```bash
mkdir -p ~/diffdock_data/data
```
### Step 3: Mount the NGC Workspace To Your Local Directory
Mount the NGC workspace to your local directory using the following command.
```bash
ngc workspace mount $WKSP_ID ~/diffdock_data --mode RW
```

### Step 4: Download Raw Data
This command downloads the PDBBind v2020 dataset from [zenodo](https://zenodo.org/records/6408497).
You can manually download it from the link or use the following command:
```bash
cd ~/ && wget https://zenodo.org/records/6408497/files/PDBBind.zip
```
Note that the data size is ~1.7GB. So, the download could take a while.
After downloading, unzip and copy the data to the mounted directory in your NGC workspace.
```bash
cd ~/ && unzip ~/PDBBind.zip
cp -r ~/PDBBind_processed ~/diffdock_data/data
```
And get the data split files from public [diffdock](https://github.com/gcorso/DiffDock), and copy it to
the mounted directory in your NGC workspace.
```bash
mkdir ~/splits
cd ~/splits
wget https://raw.githubusercontent.com/gcorso/DiffDock/main/data/splits/timesplit_no_lig_overlap_train
wget https://raw.githubusercontent.com/gcorso/DiffDock/main/data/splits/timesplit_no_lig_overlap_val
wget https://raw.githubusercontent.com/gcorso/DiffDock/main/data/splits/timesplit_test
wget https://raw.githubusercontent.com/gcorso/DiffDock/main/data/splits/timesplit_test_no_rec_overlap
cp -r ~/splits ~/diffdock_data/data
```

### Step 5: Run Pre-processing of Train Data for Score Model

This script performs pre-processing of train data for diffdock score model.

If you followed the [Quickstart Guide](./quickstart-fw.md), you may already have a .env file in your root path. If so, run `source .env` to export all the necessary variables.
If you don't have a .env file in your path, update and run the following commands to export necessary variables before running the commands below.

```
export NGC_CLI_ORG=??
export NGC_CLI_TEAM=??
export BIONEMO_IMAGE=??
```

First do protein embedding using esm2:

```bash
ngc batch run --name "DiffDock_PDBBind_Protein_Embedding" --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd 'python examples/molecule/diffdock/train.py do_embedding_preprocessing=True do_training=False '" --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```

Then, do the graph preprocessing for score model:
```bash
ngc batch run --name "DiffDock_PDBBind_Graph_Preprocessing_Score" --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd 'python examples/molecule/diffdock/train.py do_preprocessing=True do_training=False data.num_workers=20 '" --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```


### Step 6 : Verify that the data pre-processing is successful in Step 5
You can re-mount your workspace after running through all the steps above and be able to see pre-processed data. Follow Step 3 to mount the workspace onto your local workstation.'
Below is an example directory structure you should see for pre-processed diffdock training data for score and confidence models.
```bash
ls ~/diffdock_data/data
```
It will have following files:
```
data_cache
esm2_3billion_embeddings.sqlite3
PDBBind_processed
pdbbind_sequences.fasta
splits
```

```bash
ls ~/diffdock_data/data/data_cache
```

It will show following files:
```
torsion_limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings
torsion_limit0_INDEXtimesplit_no_lig_overlap_val_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings
torsion_limit0_INDEXtimesplit_test_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings
```


### Step 7: Run Pre-processing of Train Data for Confidence Model

The dataset preprocessing for confidence model will use a trained score model to do reverse diffusion and to generate ligand conformers.
It is strongly suggested to train a small score model for this to save time and to improve the accuracy of confidence model. Refer to
[Diffdock Model Training](./notebooks/model_training_diffdock.ipynb) for details of training a small score model.

Once you have a trained small score model, copy the model to the mounted directory as follow,
```
mkdir -p ~/diffdock_data/models/
cp /path/to/your/small_score_model.nemo ~/diffdock_data/models/small_score_model.nemo
export DIFFDOCK_SCORE_MODEL=models/small_score_model.nemo
```

we can do the data preprocessing for confidence model:
```bash
ngc batch run --name "DiffDock_PDBBind_Preprocessing_Confidence" --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd 'export USE_FAST_TP=1; python examples/molecule/diffdock/train.py --config-name=train_confidence do_preprocessing=True do_training=False data.num_workers=20 score_infer.restore_from_path=/workspace/bionemo/${DIFFDOCK_SCORE_MODEL} '" --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```

If you want to use the score model checkpoint converted from [public diffdock](https://github.com/gcorso/DiffDock/tree/main/workdir/paper_score_model), replace the setting of small score model with following commands:
```bash
ngc registry model download-version nvidian/clara-lifesciences/diffdock:paper_score.23.08 --dest  ~/
cp ~/diffdock_vpaper_score.23.08/diffdock_score.nemo ~/diffdock_data/models/
export DIFFDOCK_SCORE_MODEL=models/diffdock_score.nemo
```
And use the same ```ngc batch run``` command above