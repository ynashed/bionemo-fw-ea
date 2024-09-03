# BioNeMo2 Repo
To get started, please build the docker container using
```bash
./launch.sh build
```

All `bionemo2` code is partitioned into independently installable namespace packages. These live under the `sub-packages/` directory.


# TODO: Finish this.

## Downloading artifacts
Set the AWS access info in your `.env` in the host container prior to running docker:

```bash
AWS_ACCESS_KEY_ID="team-bionemo"
AWS_SECRET_ACCESS_KEY=$(grep aws_secret_access_key ~/.aws/config | cut -d' ' -f 3)
AWS_REGION="us-east-1"
AWS_ENDPOINT_URL="https://pbss.s8k.io"
```
then, running tests should download the test data to a cache location when first invoked.


## Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-fw-ea.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

### Updating pinned versions of NeMo / Megatron-LM

To update the pinned commits of NeMo or Megatron-LM, checkout that commit in the submodule folder, and then commit the
result in the top-level bionemo repository.

```bash
cd 3rdparty/NeMo/
git fetch
git checkout <desired_sha>
cd ../..
git add '3rdparty/NeMo/'
git commit -m "updating NeMo commit"
```

## Testing Locally

Inside the development container, run `./ci/scripts/static_checks.sh` to validate that code changes will pass the code
formatting and license checks run during CI. In addition, run the longer `./ci/scripts/pr_test.sh` script to run unit
tests for all sub-packages.

## Running

The following command runs a very small example of geneformer.

```bash
python  \
    scripts/singlecell/geneformer/pretrain.py     \
    --data-dir data/cellxgene_2023-12-15_small/processed_data    \
    --result-dir ./results     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2
```

## Updating License Header on Python Files
Make sure you have installed [`license-check`](https://gitlab-master.nvidia.com/clara-discovery/infra-bionemo),
which is defined in the development dependencies. If you add new Python (`.py`) files, be sure to run as:
```bash
license-check --license-header ./license_header --check . --modify --replace
```
