
# Preparing Workspace and Data for Pre-training

This section outlines the steps to prepare your workspace with pre-processed files for pre-training ESM2nv using the NVIDIA NGC platform. 

This workspace can then be used to launch ESM2nv training job using the template script `<BioNeMO_Workspace>/example/protein/esm2nv/scripts/pretrain_esm2_bcp.sh`. 

 For more information about how to launch ESM2nv training using BCP script, you can check [Running BioNeMo on DGX-Cloud using BCP](../../bcp-specific-commands-fw.md) tutorial.

### Step 1: Create a Workspace
Create a new NGC workspace by running the following command. Make sure to replace `esm2_bcp_pretrain` with your desired workspace name and `nv-us-east-2` with your preferred ACE
```bash
ngc workspace create --name besm2_bcp_pretrain --ace nv-us-east-2
```
Store the workspace ID provided after creating the workspace. You will need this ID for mounting data and running jobs.

### Step 2: Create a Local Data Directory
Create a local directory to store pre-training. For this documentation, we'll use `~/pretrain_esm2nv_data`.
```bash
mkdir ~/pretrain_esm2nv_data
```
### Step 3: Mount the Local Directory to the NGC Workspace
Mount your local data directory to the NGC workspace using the following command.
```bash
ngc workspace mount $WKSP_ID ~/pretrain_esm2nv_data --mode RW
```

### Step 4: Download Raw Data
This command downloads the pre-training dataset named `uniref50_90_202104_esm2nv`.
```bash
ngc registry resource download-version nvidian/cvai_bnmo_trng/uniref50_90_202104_esm2nv:v1.0-rc2 --dest ~/
```
Note that the data size is ~48GB. So, the download would take time (~ 7 hours)

After downloading, copy the data to the mounted directory in your NGC workspace.
```bash
cp -r ~/uniref50_90_202104_esm2nv/ ~/pretrain_esm2nv_data/
```

### Step 5: Run Pre-processing of Pre-train Data

This script performs pre-processing of pre-train data.

Set the necessary variables in your environment, specifically DATA_MOUNT, and WKSP_ID. Replace all ?? with appropriate values:

```bash
export WKSP_ID=?? # Data Workspace ID
export DATA_PATH=?? # data path in the mounted workspace
ngc batch run --name "ESM2_pretrain_preprocessing" --priority HIGH --order 1 --preempt RUNONCE --min-timeslice 0s --total-runtime 172800s --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "cd /workspace/bionemo/examples/protein/esm2nv/ && python pretrain.py do_training=False ++model.data.val_size=??  ++model.data.test_size=?? ++model.data.uf50_datapath=/data/${DATA_PATH}/uniref50_train_filt.fasta  ++model.data.uf90_datapath=/data/${DATA_PATH}/ur90_ur50_sampler.fasta   ++model.data.cluster_mapping_tsv=/data/${DATA_PATH}/mapping.tsv  ++model.data.dataset_path=/data/${DATA_PATH}/uf50 ++model.data.uf90.uniref90_path=/data/${DATA_PATH}/uf90" --result /results --image ?? --org ?? --team ?? --workspace ${WKSP_ID}:/data:RW --label ml__bionemo
```

### Step 6: Run Pre-processing of FLIP Data
If you want to enable validation in the loop of downstream tasks, you should use the following NGC command to run the pre-processing of FLIP data. 

```bash
export WKSP_ID=?? # Data Workspace ID
export DATA_MOUNT=??  # data path in the mounted workspace
ngc batch run --name "FLIP_preprocessing" --priority HIGH --order 1 --preempt RUNONCE --min-timeslice 0s --total-runtime 172800s --ace nv-us-east-2 --instance dgxa100.80g.2.norm --commandline "cd /workspace/bionemo/examples/protein/downstream && python downstream_sec_str.py do_training=False ++model.data.preprocessed_data_path=/data/${DATA_MOUNT}" --result /results --image ?? --org ?? --team ?? --workspace ${WKSP_ID}:/data:RW --label ml__bionemo
```
Ensure that you have your NGC workspace ID, relevant paths, and configurations set correctly.


