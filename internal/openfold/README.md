# openfold readme for internal use

The goal of this readme is to collect instructions useful for on-boarding
Nvidia-internal developers to the openfold project.

Before beginning developement or experimentation, a developer might
1. build and start an image
2. trigger all pytests in the repo
3. initialize initial training
4. initiailize fine-tuning

In addition to the repo-wide pytests, the openfold developer might want to
know which tests were specifically written for openfold code.  See below.

Alternatively, after steps (1) and (2) above, the developer might want to initiate
an initial training or fine tuning job.  See below.

In parallel to step 1-4 above, the developer might want to do the following procedures

* compute multiple sequence alignment

## (1) openfold-specific merge-gating tests
### (1.1) openfold-specific pytests
To run the pytests, you must first create a bionemo image and container with
standard instructions, including the step

```bash
launch.sh download_all
```
OpenFold unit tests requires nemo checkpoint and reference files from NGC.

To trigger the openfold-specific pytests, run one of the two following commands.

```bash
cd /workspace/bionemo
python download_artifacts.py --models openfold_initial_training_public openfold_finetuning_4_public openfold_initial_training_inhouse openfold_finetuning_inhouse --source pbss --model_dir $MODEL_PATH --data openfold_sample openfold_training  --verbose
pytest -k test_openfold
```

or

```bash
# inside bionemo container
cd /workspace/bionemo
python download_artifacts.py --models openfold_initial_training_public openfold_finetuning_4_public openfold_initial_training_inhouse openfold_finetuning_inhouse --\
source pbss --model_dir $MODEL_PATH --data openfold_sample openfold_training  --verbose
pytest tests/test_openfold_inference.py
pytest tests/test_openfold_checkpoint.py
```

Currently all inference tests are on sequence + MSA inputs. Inference with templates requires third party softwares and template database, and is not tested.

### (1.2) openfold-specific jet tests
There may also be JET tests, we do not yet have documentation collected for JET tests.

## (2) initial training

### (2.1) initial training data

Per conversation with `tgrzegorzek` during Feb 2024, an unoptimized version
of the openfold model was trained from random initialization to
convergence (LDDT-CA) with the dataset below.  This dataset has already
been preprocessed, with the results in `open_protein_set/preprocessed` and
`pdb_mmcif/preprocessed`.  The dataset is ~ 1 TB.

```bash
# IAD (draco) OCI cluster
/lustre/fsw/portfolios/convai/users/tgrzegorzek/data/openfold
```

This dataset has been copied (Feb 2024) to the PBSS cluster managed by
[Nvidia Core Storage](https://cssportal.sre.nsv.nvidia.com/), at the location
below:

```bash
# PBSS cluster
s3://general-purpose/datasets/openfold_from_tgrzegorzek_20240228
```

See [Bionemo Guide to PBSS](https://docs.google.com/document/d/170pjn_Gn0wFFRrLZgAwBxgmaN9ZpTXQXBWh5ZM0wBU0/edit)
for instructions on acces to PBSS.

The data copy step was estimated at ~5h based on 50MB/s transfer rate.

The dataset was copied from PBSS to the ORD, at the location

```bash
# ORD OCI cluster
/lustre/fsw/portfolios/convai/users/broland/openfold_from_tgrzegorzek_20240228
```

## (2.2) virtual pipe installation

Choose a machine as a base from which to launch `virtual pipe` jobs.  This machine
must have connectivity to the target cluster.  Call it the job-launching-machine.

1. Review 'installation' instructions at https://gitlab-master.nvidia.com/dl/JoC/virtual_pipe
2. Setup ssh connectivity between your job-launching-machine, and the cluster where compute will occur.  Make sure to ssh-add the key to the ssh agent.
3. In a conda/virtual environemnt, pip install virtual pipe per instructions above
For Ubuntu, see the instructions under Debian at [anaconda: installing on linux](https://docs.anaconda.com/free/anaconda/install/linux/)
4. Run the command `vp`, to confirm successful installation. You may have to add a path
to the `PATH` variable in your `.bashrc` or similar location.

On the job-launching-machine, the pip install command will create a directory
in `${HOME}` called `.virtualpipe` with file structure similar to

```
${HOME}
└── .virtualpipe
    ├── cluster
    │   └── draco-bcm2.yaml
    └── example
        └── slurm
            └── hello.yaml
```

When you create custom job configuration yamls, they must be put under the
`.virtualpipe` directory.

## (2.3) test a simple virtualpipe job

The command stucture to launch a virtualpipe managed job is

```bash
path-to-vp/vp --config-name path-below-virtualpipe-directory/config-filename
```

To launch and example `vp` job,

1. Update the example config yaml at .virtualpipe/example/slurm/hello.yaml:
 - Set the key '/cluster' corresponds to the target cluster.
 - Set the key 'partition' to a comma-separated listed of partition names, for your target cluster


```bash
# @package _global_

defaults:
 - /cluster: draco-bcm
 - _self_

cluster:
  walltime: 00:10:00
  docker_image: nvcr.io/nvidia/pytorch:23.06-py3
  partition: batch_singlenode
  ntasks_per_node: 1
  total_nodes: 1
  project: vp
  resumable: false
  account: convai_bionemo_training

run:
  command : nvidia-smi
  name: vp-test
```

2. Run this test job by executing this command in the virtual environment where virtualpipe
is installed

```bash
<path to vp>/vp --config-name example/slurm/hello
```

### (2.4) start an initial training job


First, we customize the contents of the `virtualpipe` job configuration yaml.

```bash
# @package _global_

defaults:
 - /cluster: draco-bcm
 - _self_

cluster:
  walltime: 04:00:00
  docker_image: gitlab-master.nvidia.com/remaining-path-to-your-docker-image
  partition: batch_block1
  ntasks_per_node: 8
  total_nodes: 16
  datasets:
    /data: /lustre/remaining-path-to-the-preprocessed-openfold-training-set
  project: openfold
  resumable: true
  account: convai_bionemo_training

run:
  command: >
    ./examples/protein/openfold/scripts/install_third_party.sh;
    python examples/protein/openfold/train.py \
      ++model.data.dataset_path=/data \
      ++trainer.num_nodes=16 \
      ++trainer.devices=8 \
      ++exp_manager.exp_dir=/result;

  name: bionemo-openfold-initial-training
```

Second, we launch the `vp` job with

```bash
path-to-vp/vp --config-name path-under-virtual-pipe/name-of-config
```

If successful, there will be stdout like

```bash
<many lines of output>
[2024-03-22 18:08:13,938][virtualpipe.backends.slurm][INFO] - Returning job id: 627807
```


Notes:
1. docker_image can also be in nvcr or gitlab container repository
2. partion names are for ORD
3. Using the dataset referenced above, no preprocessing needed. The dataset is
already pre-processed. If you would like to prepare dataset again for some
reason please have a look at `OpenProteinSet/Preprocess` and `PDBMMCIF/Preprocess` inside of `train.py`
4. In the recipe below, we (internal developers) don't use 'Initial_training_slurm.sh',
which is the public-facing scripts.

### (2.5) resume from failure [wip]

If you executed a virtualpipe command, with `resumable: True`, you should
expect to see a sequence of slurm jobs, where each ends with a message like
the following in the log file.

```bash
Apr 07 19:34:50.986548 2613564 slurmstepd   0x155551a0d700: error: *** JOB 778085 ON batch-block2-2007 CANCELLED AT 2024-04-07T19:34:50 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 122 seconds for job step to finish.
```

But, on our slurm clusters, there are many factors outside of your control,
which can lead to cluster-wide outages.  You may see your sequence-of-slurm-jobs
come to a halt, where the second-to-last slurm job fails with a message like
below, in it's corresponding log file, e.g. `slurm-%j.out`
```bash
JOBID 780444
PREVIOUS JOB 778085 FINISHED WITH TIMEOUT STATUS. RESUMING...
Apr 08 03:37:03.834962 3444994 slurmstepd   0x155552ddfd40: error: pyxis: child 3444995 failed with error code: 1
Apr 08 03:37:03.835051 3444994 slurmstepd   0x155552ddfd40: error: pyxis: failed to import docker image
Apr 08 03:37:03.835067 3444994 slurmstepd   0x155552ddfd40: error: pyxis: printing enroot log file:
Apr 08 03:37:03.835098 3444994 slurmstepd   0x155552ddfd40: error: pyxis:     [INFO] Querying registry for permission grant
Apr 08 03:37:03.835121 3444994 slurmstepd   0x155552ddfd40: error: pyxis:     [INFO] Permission granted
Apr 08 03:37:03.835130 3444994 slurmstepd   0x155552ddfd40: error: pyxis:     [INFO] Fetching image manifest list
Apr 08 03:37:03.835140 3444994 slurmstepd   0x155552ddfd40: error: pyxis:     [ERROR] Could not process JSON input
```

The last slurm job's log file will look something like
```bash
JOBID 786632
PREVIOUS JOB 780444 FINISHED WITH FAILED STATUS. RESUMING...
PREVIOUS JOB 780444 FINISHED WITH FAILED STATUS. EXIT.
```

In order to resume the sequence-of-slurm-jobs, identify the last job in
the sequence that end `DUE TO TIME LIMIT`, as in the first message in this section.
Call it `LAST_GOOD_JOB_ID`.

Additionally, locate the `launch.sub` file created by virtualpipe, and placed
in the output directory created by virtualpipe.

Submit `launch.sub` to the slurm cluster manager, with `LAST_GOOD_JOB_ID` as
the first positional argument
```bash
sbatch path-to-slurm-submission-script/launch.sub ${LAST_GOOD_JOB_ID}
```

For example
```bash
sbatch vp_out_20240331/vp20240404T1015zP_dev_br_cd-2858_validation_metric_debug_step_from_filename_20240403T1552_opt3q4q6q7q11_prbf16-mixed_n16_t8_v200_m80000_w4h_resTrue/virtualpipe_rundir_20240404_102009_448715/launch.sub 778085
```

The sequence-of-slurm-jobs will re-initiate training, with the most recent checkpoint in
`experiment_output_directory/artifacts/checkpoints`.  Here the definition of
'most recent' is determined by NeMo.

### (2.6) optimisation configurations

To activate the optimisations implemented in [m550](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/merge_requests/550/),
we add a two command-line arguments '++model.optimisations' and 'trainer.precision' as in

```bash
precision_in='32'
opt_list_in='[layernorm_triton,layernorm_inductor,mha_triton]'
python examples/protein/openfold/train.py \
  --config-name openfold_initial_training \
  ++model.data.dataset_path=/data \
  ++model.optimisations=${opt_list_in} \
  ++trainer.num_nodes=${total_nodes_in} \
  ++trainer.devices=${ntasks_per_node_in} \
  ++trainer.precision=${precision_in} \
  ...other-command-line-args...
  ++exp_manager.exp_dir=/result;
```

The possible values for `precision_in` are `32` and `bf16-mixed`.

Below, the variable `opt_list_in` is a list with any combination of the strings

```bash
mha_fused_gemm
layernorm_triton
layernorm_inductor
inductor_global
dataloader_pq
mha_triton
```

For `inductor_global` it's a little difficult to find the corresponding row in the
OpenFold HPC v3.0 roadmap tab at

[v3.0 MLPerf-Training tracker](https://docs.google.com/spreadsheets/d/1eAYJ6WqbZDXXpqDXOidvgV-2woZVusRVPaDE8rSDRUs/edit#gid=975160894)

but for the other settings, the correspondence is more clear.

## (3) fine-tuning

The workflow for fine-tuning is the same as for initial-training, but the
Pytorch Lighting training job config yaml will be different, as well as possibly
some command-line arguments.

## Clusters

```bash
Draco-oci-iad (“Virginia”):
      FAQ: https://confluence.nvidia.com/pages/viewpage.action?spaceKey=HWINFCSSUP&title=Draco-OCI-IAD+Draco+BCM+Cluster+FAQ
      Slack: #draco-oci-bcm-support

Draco-oci-ord (“Chicago”):
      FAQ: https://confluence.nvidia.com/display/HWINFCSSUP/CS-OCI-ORD+FAQ#CSOCIORDFAQ-HowdoIrequestOCIClusteraccess
      Slack: #nv-oci-ord-cs-support
Eos:
      FAQ: https://confluence.nvidia.com/display/HWINFCSSUP/Eos+HWInf+Getting+Started+Guide
      Slack: #hwinf-eos-users
```

## (a) compute muliple sequence alignment
### (a.1) identify CAMEO targets
[OpenProteinSet](https://registry.opendata.aws/openfold/) computes MSAs for CAMEO targets all the way until 2021-12-15. To obtain CAMEO targets beyond, the developer can traces all targets until 2022-07-05 under `bionemo/data/protein/openfold/reosurces/cameo_targets.json` or access directly from CAMEO webpage by specifying `start_date` following the url below.

```
https://cameo3d.org/modeling/1-week/difficulty/all/?to_date=${start_date}
```

### (a.2) set up for multiple sequence alignment
To follow the MSA generation protocol provided by public OpenFold team, The users should prepare the following MSAs.

1. mgnify_hits.ato (jackhmmer on mgnify)

2. uniref90.sto (jackhmmer on uniref90)

3. bfd_uniclust_hits.a3m (hhblits on bfd and uniclust30)

Users can download mgnify, uniref90, bfd and uniclust30 databases using download scripts from [their repo](https://github.com/aqlaboratory/openfold) or directly from PBSS.

Template search needs `hhsearch`, which is already installed in the docker, and pdb70 database.

```bash
# list databases
aws s3 ls --endpoint-url https://pbss.s8k.io general-purpose/datasets/sichu/

# download database to /data (~2TB)
aws s3 sync --endpoint-url https://pbss.s8k.io s3://general-purpose/datasets/sichu/bfd /data
aws s3 sync --endpoint-url https://pbss.s8k.io s3://general-purpose/datasets/sichu/mgnify /data
aws s3 sync --endpoint-url https://pbss.s8k.io s3://general-purpose/datasets/sichu/uniclust30 /data
aws s3 sync --endpoint-url https://pbss.s8k.io s3://general-purpose/datasets/sichu/uniref90 /data

# download database for template search
aws s3 sync --endpoint-url https://pbss.s8k.io s3://general-purpose/datasets/sichu/pdb70 /data
```

Then users can generate MSAs by specifying the path to the input fasta directory and output MSA directory. Try compute node with larger memory size if no `bfd_uniclust_hits.a3m` is returned.
```bash
#!/bin/bash
#SBATCH --account=convai_bionemo_training
#SBATCH --job-name=openfold_compute_alignment
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=batch_singlenode
#SBATCH --time=0-04:00:00

export pdbcode=$1
image=""
mounts=""

srun --nodes=1 \
        --container-image $image \
        --container-mounts $mounts \
        bash -c '''
# debugging
#set +ex

# export input and output directories
export input_dir=FASTA_DIR
export output_dir=MSA_DIR

if [ "$SLURM_PROCID" == "0" ]; then
        # clean up previous run
        subdir="${output_dir}/$pdbcode"
        if [ ! -f "$subdir/bfd_uniclust_hits.a3m" ]; then
                echo "Removing $subdir since no bfd alignment is found."
                rm -rf $subdir
        fi

        # copy fasta into temporary directory
        mkdir -p /tmp/$pdbcode
        cp /tmp/$pdbcode.fasta $input_dir/$pdbcode/

        # run alignment
        # hhsearch and pdb70_database_path are for template search only
        mkdir $output_dir
        /usr/bin/time -v python scripts/precompute_alignments.py ${input_dir}/$pdbcode $output_dir \
                --jackhmmer_binary_path=/opt/conda/bin/jackhmmer \
                --hhblits=/opt/conda/bin/hhblits \
                --uniref90_database_path=/data/uniref90/uniref90.fasta \
                --mgnify_database_path=/data/mgnify/mgy_clusters_2018_12.fa \
                --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
                --cpus_per_task 16 \
                --hhsearch=/opt/conda/bin/hhsearch \
                --pdb70_database_path=/data/pdb70/pdb70 \
                2> ${output_dir}/${pdbcode}.log
fi
'''
```

The runtime for each sequence can range from 30 minutes to above 4 hours.

### (a.3) convert from stockholm to a3m format
Currently our `infer.yaml` accepts `msa_a3m_filepaths` but jackhmmer outputs in sto format. Users can convert sto to a3m file using `convert_stockholm_to_a3m` under `bionemo/data/protein/openfold/parsers.py`.

## Appendix

- [OpenFold Engineering FAQ](https://docs.google.com/document/d/1MRbjLpKibLF9qSRUXAtAeFemP_3yERdye5pn-8RW_Bo/edit)
- https://github.com/TimoLassmann/kalign
- https://github.com/soedinglab/hh-suite
- [public OpenFold repo](https://github.com/aqlaboratory/openfold)
- [OpenProteinSet registry](https://registry.opendata.aws/openfold/)
- [CAMEO](https://cameo3d.org/)
