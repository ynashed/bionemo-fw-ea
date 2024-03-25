# openfold readme for internal use

The goal of this readme is to collect instructions useful for on-boarding 
Nvidia-internal developers to the openfold project.

Before beginning developement or experimentation, a developer might 
1. build and start an image
2. trigger all pytests in the repo

In addition to the repo-wide pytests, the openfold developer might want to 
know which tests were specifically written for openfold code.  See below.

Alternatively, after steps (1) and (2) above, the developer might want to initiate 
an initial training or fine tuning job.  See below.


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
pytest -k test_openfold
```

or 

```bash
# inside bionemo container
cd /workspace/bionemo
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

## Appendix

- [OpenFold Engineering FAQ](https://docs.google.com/document/d/1MRbjLpKibLF9qSRUXAtAeFemP_3yERdye5pn-8RW_Bo/edit)
- https://github.com/TimoLassmann/kalign
- https://github.com/soedinglab/hh-suite
