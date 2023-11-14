
# Preparing Workspace and Data for Pre-training

This section outlines the steps to prepare your workspace with pre-processed files for pre-training ESM2nv using the NVIDIA NGC platform. 

This workspace can then be used to launch ESM2nv training job using the template script `<BioNeMO_Workspace>/examples/protein/esm2nv/scripts/pretrain_esm2_bcp.sh`. 

For more information about how to launch ESM2nv training using BCP script, you can check [Running BioNeMo on DGX-Cloud using BCP](./bcp-specific-commands-fw.md) tutorial.

The commands below expect that you have NGC CLI installed on your machine. All of the commands below can then be run outside of BioNeMo Docker container.

### Step 1: Create a Workspace
Create a new NGC workspace by running the following command. Make sure to replace `esm2_bcp_pretrain` with your desired workspace name and `nv-us-east-2` with your preferred ACE
```bash
ngc workspace create --name esm2_bcp_pretrain --ace nv-us-east-2
```

Store the workspace ID provided after creating the workspace. You will need this ID for mounting data and running jobs.

```bash
export WKSP_ID=??
```

### Step 2: Create a Local Data Directory
Create a local directory to store pre-training. For this documentation, we'll use `~/pretrain_esm2nv_data`.
```bash
mkdir ~/pretrain_esm2nv_data
```
### Step 3: Mount the NGC Workspace To Your Local Directory
Mount the NGC workspace to your local directory using the following command.
```bash
ngc workspace mount $WKSP_ID ~/pretrain_esm2nv_data --mode RW
```

### Step 4: Download Raw Data
This command downloads the pre-training dataset named `uniref50_90_202104_esm2nv`.
```bash
ngc registry resource download-version nvidia/clara/uniref50_90_202104_esm2nv:1.0 --dest ~/
```
Note that the data size is ~49GB. So, the download could take a while.

After downloading, copy the data to the mounted directory in your NGC workspace.
```bash
cp -r ~/uniref50_90_202104_esm2nv_vv1.0-rc2/ ~/pretrain_esm2nv_data/
```

### Step 5: Run Pre-processing of Pre-train Data

This script performs pre-processing of pre-train data.

If you followed the [Quickstart Guide](./quickstart-fw.md), you may already have a .env file in your root path. If so, run `source .env` to export all the necessary variables.
If you don't have a .env file in your path, run the following commands to export necessary variables before running the commands below.

```
export NGC_CLI_ORG=nvidia
export NGC_CLI_TEAM=clara
export BIONEMO_IMAGE=nvcr.io/nvidia/clara/bionemo-framework:latest
```


```bash
export DATA_DIR=uniref50_90_202104_esm2nv_vv1.0-rc2 # path to where the data is inside the mounted workspace. For e.g., if your data is saved under /data/experiments/uniref50_90_202104_esm2nv_vv1.0-rc2 in your workspace, then the DATA_DIR will be experiments/uniref50_90_202104_esm2nv_vv1.0-rc2
ngc batch run --name "ESM2_pretrain_preprocessing" --priority HIGH --order 1 --preempt RUNONCE --min-timeslice 0s --total-runtime 172800s --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "cd /workspace/bionemo/examples/protein/esm2nv/ && python pretrain.py --config-name=pretrain_esm2_650M do_training=False ++model.data.uf50_datapath=/data/${DATA_DIR}/uniref50_train_filt.fasta  ++model.data.uf90_datapath=/data/${DATA_DIR}/uniref90membersandreps_ur50trainfiltreps.fasta   ++model.data.cluster_mapping_tsv=/data/${DATA_DIR}/mapping.tsv  ++model.data.dataset_path=/data/${DATA_DIR}/uf50 ++model.data.uf90.uniref90_path=/data/${DATA_DIR}/uf90" --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/data:RW --label ml__bionemo
```

### Step 6: Run Pre-processing of FLIP Data
If you want to enable validation in the loop of downstream tasks, you should use the following NGC command to run the pre-processing of FLIP data. FLIP data is automatically downloaded from a public facing [URL](http://data.bioembeddings.com/public/FLIP/fasta/)

```bash
ngc batch run --name "FLIP_preprocessing" --priority HIGH --order 1 --preempt RUNONCE --min-timeslice 0s --total-runtime 172800s --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "cd /workspace/bionemo/examples/protein/downstream && python downstream_sec_str.py do_training=False ++model.data.preprocessed_data_path=/data/FLIP" --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/data:RW --label ml__bionemo
```
Ensure that you have your NGC workspace ID, relevant paths, and configurations set correctly.

### Step 7 : Verify that the data pre-processing is successful
You can re-mount your workspace after running through all the steps above and be able to see pre-processed data. Follow Step 3 to mount the workspace onto your local workstation.'
Below is an example directory structure you should see for pre-processed ESM-2nv Pre-train data.
```
├── README.md
├── uf50
│  ├── test
│  ├── train
│  └── val
├── uf90
│  ├── cluster-map.json
│  ├── preprocessing.lock
│  ├── uf90_csvs
│  └── uf90_seqid_to_idx.json
├── mapping.tsv
├── uniref50_train_filt.fasta
├── uniref50_valid.fasta
├── uniref90membersandreps_ur50trainfiltreps.fasta
```
Below is the example directory structure you should see for downstream task pre-processed data.
```
FLIP
└── secondary_structure
    ├── test
    │   └── x000.csv
    ├── train
    │   └── x000.csv
    └── val
        └── x000.csv
```

