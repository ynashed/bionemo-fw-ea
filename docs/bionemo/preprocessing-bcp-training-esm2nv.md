# ESM2: Preparing Workspace and Data for Pre-training

This section outlines the steps to
1. Create your [NGC workspace](https://bc.ngc.nvidia.com/workspaces)
2. Download the raw pre-training dataset for ESM2nv to an NGC compute host.
3. Pre-process the raw dataset on an NGC compute host, and copy the
pre-processed dataset to your NGC workspace.

For these steps, we use the NVIDIA NGC platform.  The NGC workspace provides
persistent storage for the pre-processed dataset.

Your NGC workspace, and ESM2nv pre-processed dataset, support an ESM2nv
pre-training job using the template script `<BioNeMo_Workspace>/examples/protein/esm2nv/scripts/pretrain_esm2_bcp.sh`.  This script can be configured to consume the
pre-processed dataset directly from the NGC worskpace.

For more information about how to launch ESM2nv training using a BCP script,
you can check [Running BioNeMo on DGX-Cloud using BCP](./bcp-specific-commands-fw.md)
tutorial.

The commands below expect that you have the NGC CLI [installed](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#installing-ngc-cli) and [configured](https://docs.nvidia.com/base-command-platform/user-guide/latest/#configuring-the-cli-for-your-use)
on your machine.  All of the commands below can then be run outside of a BioNeMo Docker
container.  Your machine serves as a terminal for submitting NGC commands.

NOTE: If this is your first time, make sure to follow the instructions on the [pre-requisites documentation](./pre-reqs.md) page!

## Part 1: Create a Workspace
### Step 1.1: Add Variables to Your Environment
From a shell on your machine, run the commands below to
1. override NGC settings established in the step [configure](https://docs.nvidia.com/base-command-platform/user-guide/latest/#configuring-the-cli-for-your-use)
2. provide arguments to `ngc` commands below.
```bash
# shell on your machine
export NGC_CLI_ORG=nvidia
export NGC_CLI_TEAM=clara
export BIONEMO_IMAGE=nvcr.io/nvidia/clara/bionemo-framework:latest
export ACE=nv-us-east-2
```

Note:
1. If you followed the [Quickstart Guide](./quickstart-fw.md), you may already
have an .env file set up with the needed variables above, except for ACE. If so,
run `source <path to bionemo repo>/.env`, and then run the assignment
statement for ACE above.  Make sure to replace `nv-us-east-2` with your
preferred ACE.
2. The NGC_CLI_ORG value is `nvidia`, and the prefix appearing in the name of
the docker image is also `nvidia`.
3. Your NGC CLI api key will be available to `ngc` command invocations below, if you've
[configured](https://docs.nvidia.com/base-command-platform/user-guide/latest/#configuring-the-cli-for-your-use) the NGC CLI.

### Step 1.2: Command to create a workspace
Create a new NGC workspace by running the following commands. Make sure to
replace `my_ngc_workspace` with your desired workspace name.

```bash
# shell on your machine
NGC_WORKSPACE_NAME=my_ngc_workspace
ngc workspace create --name "${NGC_WORKSPACE_NAME}" --ace "${ACE}" --org nvidian
```

Store the workspace ID provided after creating the workspace in a location
on your machine. You will need this ID for mounting data and running jobs.
Also, run the below to add the workspace ID to your environment for Part 2
and 3.

```bash
export WKSP_ID=??
```

### Step 1.3: Mount the NGC Workspace To a Directory on Your Machine
Mount the NGC workspace to a directory on your machine using the following
command.
```bash
mkdir ~/ngc_workspace_mount
ngc workspace mount "${WKSP_ID}" ~/ngc_workspace_mount --mode RW --org nvidian
```
Note:
1. Data transfer between the NGC workspace and your machine's hard-drive can be
slow, so we do not recommend it.
2. With this mount, you can issue commands, like `ls` and `du` from your
machine's os to explore the contents of your workspace.
3. You can choose another location in your machine's filesystem to serve as
the mount point.

## Part 2: Download the Dataset

### Step 2.1: Start an Interactive Job on an NGC Compute Instance
With the variables above in the environment on your machine, run the command
below to submit a request to NGC achieving
1. Allocate a compute instance
2. Start a docker container with the `BIONEMO_IMAGE`
3. Mount your NGC workspace to the container at `/ngc_workspace`

```bash
# shell on your machine
INSTANCE_TYPE=dgxa100.80g.1.norm
ngc base-command job run --name "my_interactive_session" \
    --priority HIGH \
    --order 1 \
    --preempt RESUMABLE \
    --min-timeslice 0s\
    --total-runtime 86400s \
    --ace "${ACE}" \
    --instance "${INSTANCE_TYPE}" \
    --image "${BIONEMO_IMAGE}" \
    --org "${NGC_CLI_ORG}" \
    --team "${NGC_CLI_TEAM}" \
    --workspace "${WKSP_ID}:/ngc_workspace_mount:RW" \
    --result /results \
    --commandline "sleep infinity"
```
If successfull, you will see output like
```bash
----------------------------------------------------------
 Job Information
   Id: job-id-with-7-digits
   Name: my_interactive_session
   Number of Replicas: 1
```
Store the `Id` to a shell variable
```bash
export JOB_ID=job-id-with-7-digits
```
In a separate shell instance, submit a request to NGC to start a
shell in the NGC-hosted container
```bash
# shell in your machine
ngc batch exec "${JOB_ID}"
```
If successfull, you'll see a command prompt like
```bash
# shell in the NGC-hosted container
root@your-job-id:/workspace#
```
Please run the command below to adjust terminal:
```
resize
```
Note:
1. The command-line environment in the NGC-hosted BIONEMO_IMAGE container is
very minimal.  If you notice the terminal behaving unusually, you can try
running `resize` again.
2. An NGC job, configured as above, will terminate after a duration equal
to `total-runtime`, or roughly 24h after creation, whichever is earlier.
There is a runtime cutoff at 24h, not controlled by the user.  The contents
of your NGC workspace storage will persist.
3. In the separate shell instance, NGC CLI credentials will be provided from the
the [configuring](https://docs.nvidia.com/base-command-platform/user-guide/latest/#configuring-the-cli-for-your-use) step.


### Step 2.2: Configure NGC client on NGC host
From your shell on the NGC host, configure the NGC client with the values of
your NGC org, team, api key, and ace, by running
```bash
ngc config set
```
and enter your NGC user information.  For the setting `format type`,
the setting `ascii` is typical.

If successfull, you will see output similar to
```bash
Successfully saved NGC configuration to /root/.ngc/config
```

Note:
1. You can check your settings with `ngc config current`.

### Step 2.3: Download Raw Data
In this step, we download the raw data to the hard-drive local to your
NGC compute instance, not to your NGC workspace storage.

First, let's confirm that your NGC client is configured to have visibility
to the pre-training dataset named  `uniref50_90_202104_esm2nv`.
```bash
DATASET_NAME=nvidia/clara/uniref50_90_202104_esm2nv
DATASET_DIR_NAME=uniref50_90_202104_esm2nv_v1.0
ngc registry resource list "${DATASET_NAME}"
```
If successful, you will see output similar to:

```bash
root@5639990:/workspace# ngc registry resource list $DATASET_NAME
+------------------------+--------------------------+----------------+-------------+-----------+-----------+---------------+------------+
| Name                   | Repository               | Latest Version | Application | Framework | Precision | Last Modified | Permission |
+------------------------+--------------------------+----------------+-------------+-----------+-----------+---------------+------------+
| ESM2nv Uniref Training | nvidia/clara/uniref50_90 | 1.0            | OTHER       | Other     | OTHER     | Nov 28, 2023  | unlocked   |
| Data                   | _202104_esm2nv           |                |             |           |           |               |            |
+------------------------+--------------------------+----------------+-------------+-----------+-----------+---------------+------------+
```

The following command downloads the pre-training dataset named
`uniref50_90_202104_esm2nv` into the local directory
`/workspace/data_from_ngc` as a subdirectory `uniref50_90_202104_esm2nv_v1.0`.


```bash
# shell in NGC-hosted container
mkdir /workspace/data_from_ngc
ngc registry resource \
    download-version "${DATASET_NAME}" \
    --dest /workspace/data_from_ngc
```
If successfull, you will see output similar to
```
Getting files to download...
⠙ ╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ • 0.6/48.9 GiB • Remaining: 0:18:33 • 46.7 MB/s • Elapsed: 0:00:14 • Total: 5 - Completed: 2 - Failed: 0
```

Note:
1. Time estimate: < 1h.  The data size is ~49GB.
2. The `/workspace directory` is file-system space local to the compute instance,
and is intended to be your sandbox, but only for the duration of the NGC
interactive job.
3. We download to the NGC compute instance, rather than your machine, to
use the fast data transfer between the NGC compute instance and the NGC
workspace.
4. The name of the dataset `DATASET_NAME` is different from the name of
download subdirectory name `DATASET_DIR_NAME`.  Please check the name of the
subdirectory after download and update the environment variable `DATASET_DIR_NAME`.

### Step 2.4: Copy Raw Dataset to your NGC Worksapce
After downloading, copy the dataset from the NGC compute instance to the NGC
workspace, for use exterior to the current NGC interactive session
```bash
# shell in NGC-hosted container
mkdir /ngc_workspace_mount/data_from_ngc
cp -r "/workspace/data_from_ngc/${DATASET_DIR_NAME}" \
    /ngc_workspace_mount/data_from_ngc/
```
Note:
1. Time estimated: < 1 min.
2. We copy to the workspace, so that you may split your work session over
multiple days, which will require a separate interactive NGC compute job.

At this point, you can continue to Part 3 with your NGC compute job.  Alternatively,
you can terminate the NGC job created in Step 2.1, and do Part 3 with a separate
NGC compute job.

## Part 3: Preprocess the Raw Dataset
Part 3 can be done in the same interactive session as Part 2, or in a separate
interactive session.  The raw dataset will be obtained from the copy on
your workspace.

### Step 3.1:  Setup
To start an interactive job with an NGC compute instance, repeat Step 1.1 from
your machine and set

```bash
export WKSP_ID=??
```
and then repeat Step 2.1

### Step 3.2: Pre-process the Pre-train Data
In this step, we'll run a pre-processing bash script that
1. Copies the raw dataset from your NGC workspace to the local hard-drive on
the ngc compute instance.
2. Runs the preprocesing task
3. Copies the pre-processed back to your NGC workspace.

From the shell in the NGC-hosted container, run

```bash
# shell in NGC-hosted container
export DATASET_DIR=uniref50_90_202104_esm2nv_v1.0  # subdirectory created in Part 1
export DATA_DIR_IN_REMOTE_HD=/ngc_workspace_mount/data_from_ngc
export DATA_DIR_IN_LOCAL_HD=/workspace/data_from_ngc
export SCRIPT_DIR="${BIONEMO_HOME}/examples/protein/esm2nv/scripts"
cd /workspace
"${SCRIPT_DIR}/preprocess_esm2_bcp.sh" \
    &> "/ngc_workspace_mount/preprocess_esm2_bcp_$(date +'%Y%d%mT%H%M%S').log" &
```

The output pre-processed dataset location is written at the end of the log file,
and will have the form `/ngc_workspace_mount/data_from_ngc/${DATASET_DIR}_preproc`.

Note:
1. Time estimate: < 1h.
2. You can monitor the progress (and view possible errors) of the
preprocessing task from your machine (not the NGC host), with your
NGC workspace mounted, by opening the log file in a text editor, or with the
linux command `tail`.
3. From the shell in the NGC-hosted container, the variable `BIONEMO_HOME`
should be set by default, you do not have to set it.  The expected value
is `/workspace/bionemo`.
4. The script `preprocess_esm2_bcp.sh` does not require `NGC_CLI_ORG`,
`NGC_CLI_TEAM`, or `NGC_CLI_API_KEY` in the calling shell environment.

### Step 3.3: Run Pre-processing of FLIP Data
If you want to enable validation in the loop of downstream tasks, you should
pre-process the FLIP data. FLIP data is automatically downloaded from a public
facing [URL](http://data.bioembeddings.com/public/FLIP/fasta/)

From the shell in the NGC-hosted container, run

```bash
# shell in NGC-hosted container
cd /workspace
python "${BIONEMO_HOME}/examples/protein/downstream/downstream_flip.py" \
    do_training=False \
    ++model.data.preprocessed_data_path=/workspace/data_from_ngc/FLIP
cp -r /workspace/data_from_ngc/FLIP "/ngc_workspace_mount/data_from_ngc/"
```
Ensure that you have your NGC workspace ID, relevant paths, and configurations
set correctly.

Note:
1. From the shell in the NGC-hosted container, the variable BIONEMO_HOME should
be set by default, you do not have to set it.

### Step 3.4: Verify that the data pre-processing is successful
On your machine, view the contents of the output data directories on the
mounted NGC workspace. If needed, follow Step 1.3 to re-mount the workspace
onto your NGC-CLI-installed machine.

For example
```bash
# shell on your machine
ls ~/ngc_workspace_mount/data_from_ngc/uniref50_90_202104_esm2nv_v1.0_preproc
```
Below is an example directory structure you should see for pre-processed
ESM-2nv Pre-train data.

```
├── README.md
├── uf50
│  ├── test
│  ├── train
│  └── val
├── uf90
│  └── uf90_csvs
├── mapping.tsv
├── uniref50_train_filt.fasta
├── uniref50_valid.fasta
├── uniref90membersandreps_ur50trainfiltreps.fasta
```
For the FLIP data
```bash
# shell on your machine
ls ~/ngc_workspace_mount/data_from_ngc/FLIP
```
should show a directory structure like

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

Note:
1. You may observe that the `ls` command is slower when applied to the NGC
workspace mounted to your machine, than when applied to a directory local to
your machine's hard drive.
2. The subdirectory name `uniref50_90_202104_esm2nv_v1.0_preproc` may
need to be adjusted so that the prefix matches the value of `DATASET_DIR_NAME`,
see Note 4 in Step 2.3.
