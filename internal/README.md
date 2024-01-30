# Quickstart Guide
## 0. Get the docker image:

`devel` docker image is created in the CI for:
* every merge to `dev` (automatically)
* every MR (if you trigger the pipeline)

The image are tagged with `<base container tag>-devel`, so the tags are:
* `gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel`
* `nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-devel` (same image as above)
* `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel`

See [Building devel docker image manually](#building-devel-docker-image-manually) for instructions on how to build your own image.

## 1. Launch `devel` container for interactive development:

```bash
BIONEMO_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel"  # from dev
# BIONEMO_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel"  # from your MR
echo "BIONEMO_IMAGE=${BIONEMO_IMAGE}" >> .env  # or edit .env file manually
docker pull ${BIONEMO_IMAGE}  # make sure it's up to date
./launch.sh dev
```

## 2. Run performance linter:
```bash
bash $BIONEMO_HOME/internal/run_tpl.sh /path/to/your/script.py arg1=val1 +arg2=val2 ...
```

## 3. Run profilers:
```bash
TAP_MODE=auto python /path/to/your/script.py arg1=val1 +arg2=val2 ...
```

# Performance analysis and optimization

`devel` docker container contains two tools that can help you to analyze and optimize performance of your model:
* [Torch Performance Linter](https://gitlab-master.nvidia.com/dl/gwe/torch_performance_linter) which analyzes your code and detects the most common inefficiencies (like tensor shapes not divisible by 8)
* [Torch Automated Profiler](https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler) which wraps your code with different profilers (Nsight, DLProf, ANNA, Kineto)

You should run these tools with short experiments, so limit dataset size and number of steps. Overrides to your default config might look like this:

```
++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2
```

Don't limit the number of steps too much to give all profilers enough steps to collect data.

Note, that profiling results are highly hardware-dependent, so make sure to run TAP on a target device (eg A100) as the final check (running on your workstation is a good starting point).


To analyze your workload with TPL, replace `python` in your command with `bash $BIONEMO_HOME/internal/run_tpl.sh`, eg:
```bash
overrides="++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2"

bash $BIONEMO_HOME/internal/run_tpl.sh /workspace/bionemo/examples/protein/prott5nv/pretrain.py ${overrides}
```

This will print information about all detected issues, the overall score for your code, a summary with issues grouped together, and path to full log. The output will look similar to this:
```bash
TPL: #3990 Torch Performance Linter Score: 8.32/10
--------------------------------- SUMMARY ---------------------------------
priority category                      name count                                                                                                                                                                                                                                    msg
      P0      AMP              AMP_AUTOCAST     6                                                                                                                                                                                PyTorch AMP (torch.cuda.amp.autocast) should be enabled
      P0     DATA          SYNC_DATA_LOADER     1                                                                                                                             use asynchronous torch.utils.data.DataLoader with num_workers > 0, detected a call with {'num_workers': 0}
      P1      CPU           OMP_NUM_THREADS     1                                                                                               detected env variable OMP_NUM_THREADS=None, set OMP_NUM_THREADS, the optimal value depends on the workload, default to OMP_NUM_THREADS=1
      P1     DATA NONPERSISTENT_DATA_LOADER     1                                                                                                                       use torch.utils.data.DataLoader with persistent_workers=True, detected a call with {'persistent_workers': False}
      P1      GPU              L2_PROMOTION     1                                                                                                                      increase maximum fetch granularity of GPU L2 cache to 128 bytes, detected cudaLimitMaxL2FetchGranularity=64 bytes
      P2      VER             TORCH_VERSION     1                                                                                                                                             minimum recommended PyTorch version is 2.0.0, currently running in version 2.0.0a0+1767026
       W  COMPUTE         TENSOR_CONTIGUOUS   408                                                                                                called torch.Tensor.contiguous on non-contiguous Tensor(device=cuda:0), data transposes are expensive and should be avoided if possible
       W  COMPUTE        UNSPECIFIED_DEVICE   127                                           called torch.empty with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W  COMPUTE        UNSPECIFIED_DEVICE   119                                          called torch.tensor with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W  COMPUTE        UNSPECIFIED_DEVICE     1                                           called torch.zeros with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W     EVAL         EVAL_GRAD_ENABLED   288 call to a nn.Module in eval mode returned a tensor with an attribute requires_grad=True, typically gradients are not needed in evaluation mode, run the evaluation under torch.no_grad context manager to disable gradient computation
       W      GPU               EMPTY_CACHE     3                                                                                                                                                               Forcing a cache clear can be expensive and should be avoided if possible
       W     SYNC          CUDA_SYNCHRONIZE     1                                                                                                                                          torch.cuda.synchronize() causes host-device synchronization and should be avoided if possible
       W     SYNC               TENSOR_ITEM     8                                                         called torch.Tensor.item on Tensor(device=cuda:0), torch.Tensor.item causes an implicit host-device synchronization if called on GPU tensors and should be avoided if possible
       W     SYNC               TENSOR_ITEM     4                                                            called torch.Tensor.item on Tensor(device=cpu), torch.Tensor.item causes an implicit host-device synchronization if called on GPU tensors and should be avoided if possible
---------------------------------------------------------------------------
See full log in torch_performance_linter_3990.log
```


To analyze your workload with TAP, add `TAP_MODE=auto` before your command, eg:

```bash
overrides="++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2"


TAP_MODE=auto python /workspace/bionemo/examples/protein/prott5nv/pretrain.py ${overrides}
```

TAP will create a dir, by default `tap_results`, containing generated profiles. The easiest way to visualize the results is to use a dedicted container:
```bash
docker run -v <local path to TAP results>:/tap_results -p 8000-8003:8000-8003 -t gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler/viz:release
```
Then open localhost:8000 in your browser and select the profile you want to see.


For more details and instructions how to visualize and interpret the results, see
* https://gitlab-master.nvidia.com/dl/gwe/torch_performance_linter
* https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler
* GWE slack channel: #ghw-dl-gwe


# Building devel docker image manually

Building image manually is useful if:
* you introduced changes in `internal` dir locally
* you work with a custom BioNeMo build and you want `devel` version for it

If you have an MR with all your changes, you can use CI built image instead - it is created automatically for every pipeline and tagged with: `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel`


`devel` image is built on top of the BioNeMo image and is defined with `internal/Dockerfile-devel`.
Building this image requires you to provide `BIONEMO_IMAGE` argument, which specifies the base docker image:
```
--build-arg BIONEMO_IMAGE=<docker image tag>
```

This base image can be created by you locally with `./launch.sh build` or taken from our registry, eg:
* `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<CI pipeline ID>` to extend docker image created from your MR
* `gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest` to extend latest image created from `dev` branch.


As we install internal tools, you need to be connected to VPN and set up a ssh agent before building to authenticate:

```bash
eval "$(ssh-agent -s)"
find ~/.ssh/ -type f -exec grep -l "PRIVATE" {} \; | xargs ssh-add &> /dev/null
```

Then you can build the docker image as follows:
```bash
BASE_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest
docker build --network host --ssh default --build-arg BIONEMO_IMAGE="$BASE_IMAGE" -t ${BASE_IMAGE}-devel -f internal/Dockerfile-devel .
```

# Running tests of BioNeMo's models in JET

Running tests of the models in JET is useful since it provides a unified E2E framework for trainings, evaluations or testing.
It consists of the following steps
* getting BioNeMo's container
* running a test's script on a predefined type of machines inside the container,
* collecting information about the tests in a standardised way by logging predefined metrics
* providing unified output information about the test results

The tests in JET should be simple and short training or testing commands similar to `examples/tests/conf` that are executed in a BioNeMo's docker container across a grid of parameters
such as different
* number of nodes: 1, 4
* number of gpus: 1, 8
* precisions: 16-mixed, bf16-mixed, 32
* batch sizes: two model-specific batch sizes ie 32, 64

that are supplied to the python command. Therefore, by default, each python command for training or testing is run across `2^4` combinations of these parameters that results in `2^4` tested cases of a python training command.
The configs of the tests that are supplied to JET CI can be found in JET Workloads Registry, ie check reference of the tests that are used in BioNeMo CI for unit testing:
[bionemo/training-unit-test](https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/tree/bionemo/training-unit-tests/recipes)

## Running JET tests for a single BioNeMo model from the command line

The tests in JET can be only run for a model that is implemented in a specified BioNeMo's container.
The python method `internal/jet/run_jet_downstream_pipeline.py` allows to run the test for a single model from the command line.
It requires [JET API](https://gitlab-master.nvidia.com/dl/jet/api) to be installed. It uses as a default test template config, the JET reference [bionemo/new-model-dev](https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/blob/bionemo/new-model-dev/recipes/config.yaml) with default container set to `dev-latest-devel`.
During the testing, models checkpoints are downloaded to NGC under `/model` in the docker container by executing `python download_models.py all --download_dir $MODEL_PATH` before the python script.

### Defining tests
See the docstrings of `internal/jet/run_jet_downstream_pipeline.py` for more info about its arguments. The most important ones for building a training script and specifying tested cases are:
* for the training script:
```bash
config_path: str, path to the folder with primary training/testing configs, relative to $BIONEMO_HOME
config_name: str, hydra config name to use that can be found under config_path
script_path: str, path to a folder with training/testing scripts to use, relative to $BIONEMO_HOME
variant: str, name of a training/testing script to use (without .py extension), relative to the script_path
model: str, name of the model to be tested
extra_overwrites: str, additional training configs to be passed
```
The created partial training command in JET using those arguments is executed under `$BIONEMO_HOME` in a BioNeMo's docker container
and is of the following form
```bash
python <script_path>>/<variant>.py --config-path {BIONEMO_HOME}/<config_path> --config-name <config_name> \
<extra_overwrites> ...
```

* specifying tested cases
```bash
dllogger_warmup: int, warmup steps of DLLOGGER before the metrics are logged
nodes: int or a sequence of ints seperated by separated by space, number(s) of nodes to be tested
gpus: int or a sequence of int seperated by separated by space, number(s) of devices (gpus) to be tested
precision: int or a sequence of ints seperated by separated by space, precision(s) to be tested
batch_size: int or a sequence of ints seperated by separated by space, batch size(s) to be tested
```
and the resulting partial training command is the following for `<NODE> in <nodes>`, `<GPU> in <gpus>`, `<PRECISION> in <precisions>`
```bash
python <BIONEMO_HOME>/<SCRIPOT_PATH>/<VARIANT>.py --config-path <BIONEMO_HOME>/<CONFIG_PATH> --config-name <CONFIG_NAME> \
<ADDITIONAL_TRAINING_CONFIGS> trainer.num_nodes=<NODE> trainer.devices=<GPU> trainer.precision=<PRECISIONS>  \
model.micro_batch_size=<BATCH_SIZE> ...
```
As mentioned in the introduction, since we test each training case do all combinations of values in arrays `<nodes>`, `<gpus>`, `<precisions>` and <batch_sizes>
the resulting number of tested cases is `|<nodes>|x|<gpus>|x|<precisions>|x|<batch_sizes>|`

#### Running JET tests for a single BioNeMo model using `run_jet_downstream_pipeline.py`

The related command using `internal/jet/run_jet_downstream_pipeline.py` runs tests in JET based on the config in JET Workloads Registry under [bionemo/new-model-dev](https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/blob/bionemo/new-model-dev/recipes/config.yaml).
The default docker that is used for this test is `dev-latest-devel` and the command to run the test is

```bash
python internal/jet/run_jet_downstream_pipeline.py --model <MODEL_NAME> --config_path <CONFIG_PATH> --config_name <CONFIG_NAME> --script_path <SCRIPT_PATH> --variant <VARIANT> --nodes <NODE1> <NODE2>
--gpus <GPU1> <GPU2> --precision <PREC1> <PREC2> --batch_size <BATCH_SIZE1> <BATCH_SIZE2>
--extra_overwrites <ADDITIONAL_TRAINING_CONFIGS>
```
for instance, given a test training for `MegaMolBART`

```bash
python internal/jet/run_jet_downstream_pipeline.py --model megamolbart --config_path examples/molecule/megamolbart/conf \
--config_name pretrain_xsmall_span_aug --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 \
--gpus 1 8 --batch_size 4 8 --precision 16 32 --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=null ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"
```
Once again, let's remark that tests are executed under `$BIONEMO_HOME` in a container.
The argument `--extra_overwrites` can be supplied with all additional configurations that is needed to specify your training or testing script
See `internal/jet/scripts/toy_example_run_jet_pipeline_new_model.sh` for an example of a testing pipeline in JET.

### Running test for the first time
In order to run tests for the first time, JET API credentials need to be set. There are two ways how it can be set
* by appending flag `--setup_jet_api` when `python internal/jet/run_jet_downstream_pipeline.py ...` is executed for the first time
* or by directly calling `jet secrets login;jet secrets pull` from the command line.

Also, make sure your git is not configured as well as its credentials as lack of this information may cause the command to throw errors

### Referring to non-primary configs by hydra-searchpath
Accessing non-primary configs via hydra-searchpath can be done by `--extra_overwrites`
See an example of unit test training for MegaMolBART given config from `examples/tests/conf/`
```bash
python internal/jet/run_jet_downstream_pipeline.py --model megamolbart --config_path examples/tests/conf --config_name megamolbart_test \
 --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 --gpus 1 --batch_size 8 --precision 16 \
 --extra_overwrites "hydra.searchpath=[file://$BIONEMO_HOME/examples/molecule/megamolbart/conf] ++model.dwnstr_task_validation.enabled=False"
```

### Supplying or building BioNeMo docker image for tests in JET
Two ways of specifying how to get a docker image for the tests are available in `internal/jet/run_jet_downstream_pipeline.py`:
* a precise name of the docker image to be used and is supplied by `--image` ie

`--image gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel` (default in the config)
  ```bash
  python internal/jet/run_jet_downstream_pipeline.py --image "gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel" \
  --model megamolbart --config_path examples/molecule/megamolbart/conf --config_name pretrain_xsmall_span_aug \
  --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 --gpus 1 --batch_size 8 --precision 16 \
   --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=null ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"
  ```
* details of a git repositiory, git branch and path to `Dockerfile` to build a container during the JET pipeline by supplying `--git_repo --git_branch --dockerfile` ie `--git_repo https://gitlab-master.nvidia.com/clara-discovery/bionemo.git --git_branch dev  --dockerfile setup/Dockerfile`
```bash
python internal/jet/run_jet_downstream_pipeline.py --git_repo "https://gitlab-master.nvidia.com/clara-discovery/bionemo.git" \
--git_branch dev  --dockerfile setup/Dockerfile  --model megamolbart --config_path examples/molecule/megamolbart/conf \
--config_name pretrain_xsmall_span_aug --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 --gpus 1 \
--batch_size 8 --precision 16 --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=null ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"
 ```

### Running tests for BioNeMo's models present in `dev`
The default docker image that is used for tests defined by [bionemo/new-model-dev](https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/blob/bionemo/new-model-dev/recipes/config.yaml) is `dev-latest-devel`.
Hence, to tests a model which implementation is available in this docker, there is no need to supply neither a docker image  (`--image`) nor args that specify how to build a docker from the git branch
(`--git_repo, --git_branch, --dockerfile`). For instance,
```bash
python internal/jet/run_jet_downstream_pipeline.py --model megamolbart --config_path examples/molecule/megamolbart/conf \
--config_name pretrain_xsmall_span_aug --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 --gpus 1 \
--batch_size 8 --precision 16 --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=null ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"
```

### Running tests during a new model development
It is important to consistency check a new model performance during its development. Running JET tests for a single model
is an easy-to-use and convenient tool that provides with this information. If one has a docker image with the latest code related to the new model, it can be supplied directly
to the related command

`python internal/jet/run_jet_downstream_pipeline.py --image <DOCKER_NEW_MODEL> ...`. For instance,

```bash
python internal/jet/run_jet_downstream_pipeline.py --image gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel ....
```

Most likely, there is no most up-to-date docker image and building it just for the JET tests is very cumbersome and a time-consuming task.
As mentioned in the previous section, we can supply information on how to build a docker from the git repository and it will be build during the JET pipeline.
Supply `--git_repo` (link of the git repo with the code related to the new model), `--git_branch` (the related branch name) and `--dockerfile` (path to the `Dockerfile` that is used to build a docker for testing, relative to `$BIONEMO_HOME`).
For instance,
```bash
python internal/jet/run_jet_downstream_pipeline.py --git_repo "https://gitlab-master.nvidia.com/clara-discovery/bionemo.git" \
--git_branch dev  --dockerfile setup/Dockerfile ....
```
It is encourage to test a new model supplying
* number of nodes: 1, 4
* number of gpus: 1, 8
* precisions: 16, 32
* batch sizes: two model-specific batch sizes (a first one that saturates GPU and the second one that equals the first one divided by 2)

### Rerunning tests for reference that exists in JET Workloads Registry
If one wants to rerun an existing JET test, referenced by `JET_REF`, it can be achieved by executing
```bash
python internal/jet/run_jet_downstream_pipeline.py --jet_ref <JET_REF>
```
`JET_REF` is a valid reference when it is present as a branch in [JET Workloads Registry](https://gitlab-master.nvidia.com/dl/jet/workloads-registry) ie it is an official test used in BioNeMo CI or a temporary test defined as `ephemeral/bionemo/*` branch (their definition is present only up to 2 week after running a test).

To check the validity of the reference, execute
```bash
jet workloads --registry-ref <JET_REF> list
```

### Getting information about the tests
Use method `internal/jet/get_results_from_jet.py` to query the results of BioNeMo's tests in JET.
The results can be get for all BioNeMo's tests, only for specific reference or pipeline or workload ids.
They can be saved to a .csv file or printed to the console (default)


* Get results for all tests run for BioNeMo account (main tests and the ones run on ephemeral/bionemo), ie
```bash
python internal/jet/get_results_from_jet.py
```
* Get reference-related results, ie
```bash
python internal/jet/get_results_from_jet.py --jet_workloads_ref bionemo/training-unit-tests
```

* Get pipeline id-related results
```bash
python internal/jet/get_results_from_jet.py --pipeline_id 9469458
```
* Get results for tests based on reference pattern, ie ephemeral branches
```bash
python internal/jet/get_results_from_jet.py --jet_workloads_ref_pattern "ephemeral/bionemo/.*"
```
See `internal/jet/scripts/toy_example_run_jet_pipeline_new_model.sh` for an example of the testing pipeline in JET.



#### Choices of displaying information about the tests
Attach to your command `--print_script` to print commands of the failed tests to the console and related docker info
to replicate the errors
`python internal/jet/get_results_from_jet.py ..... --print_script`

Or attach to your command `--save_dir <FOLDER_PATH>` to save details of the run tests to a .csv file with predefined path

`python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>`

When no .csv file is provided by `--save_dir`, the text printed do the command contains information about docker build such as
* a docker image name or
* a recipe how a docker was build from git repository (git repo name, branch and path to the Dockerfile)
```bash
2023-08-21 16:55:49,043 - DOCKER BUILD build/bionemo with status: SUCCESS
JET Workloads ref: ephemeral/bionemo/9408502, JET pipeline id: 9410619, JET job id: 66976930, JET workload id: ffbb05accb0d4e7c8bca7c19f7bcd10a

Docker info:
s_image: gitlab-master.nvidia.com:5005/clara-discovery/bionemo:pipeline-9408502
```
and information about success of the tests and relevant details such as
* details of the JET pipeline and jobs (JET Workload ref, pipeline id, job id and workload id)
* link to training logs (Dllogger for performance or convergence or other metrics)
* link to output script logs (for easy access to the information what went wrong)
* script of the commands that failed and related docker info to replicate results (if `--print_script`)

```bash
2023-08-21 17:18:40,853 - PERFORMANCE TEST recipe/megamolbart_pretrain_bionemo-jet_perf-train_32_nodes-1_gpus-1_bs-32 with status: SUCCESS
JET Workloads ref: dorotat/bionemo-perf-poc, JET pipeline id: 8307367, JET job id: 60422651, JET workload id: aaa49cdb27f54026a21bc818736fc70c
dllogger: https://pbss.s8k.io/v1/AUTH_team-jet-logs/aaa49cdb27f54026a21bc818736fc70c/dllogger.json
output_script: https://pbss.s8k.io/v1/AUTH_team-jet-logs/aaa49cdb27f54026a21bc818736fc70c/output_script.log

2023-08-21 16:55:49,043 - PERFORMANCE TEST recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false with status: FAILED
JET Workloads ref: ephemeral/bionemo/9408502, JET pipeline id: 9410619, JET job id: 66976957, JET workload id: e2bcb1647606421b9fb05e3797fe31d9
output_script: https://pbss.s8k.io/v1/AUTH_team-jet-logs/e2bcb1647606421b9fb05e3797fe31d9/output_script.log

Script:
cd $BIONEMO_HOME
export NGC_CLI_ORG=nvidian
export NGC_CLI_TEAM=clara-lifesciences
export NGC_CLI_FORMAT_TYPE=ascii
export MODEL_PATH=/model
python download_models.py all --download_dir ${MODEL_PATH} --verbose
python examples/molecule/megamolbart/pretrain.py --config-path $BIONEMO_HOME/examples/tests/conf/ --config-name megamolbart_test \
trainer.num_nodes=1 trainer.devices=8 trainer.precision=32 model.micro_batch_size=32 \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.create_wandb_logger=False ++exp_manager.create_tensorboard_logger=False ++exp_manager.exp_dir=/tmp/nemo_experiments/ ++exp_manager.create_checkpoint_callback=False ++exp_manager.resume_if_exists=False  hydra.searchpath=[file://$BIONEMO_HOME/examples/molecule/megamolbart/conf] \
++create_dllogger_callbacks=True ++create_trainer_metric_callback=True \
++dllogger_callbacks_kwargs.use_existing_dllogger=True ++dllogger_callbacks_kwargs.warmup=100 \
++dllogger_callbacks_kwargs.json_file=/jet/logs/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false/dllogger.json ++trainer_metric_callback_kwargs.log_path=/jet/assets/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false ++logs_dir=/jet/logs/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false
```

## Running JET tests in BioNeMo CI for models in BioNeMo
The automatic testing in JET is enabled in BIoNeMo's CI. It's primary goal is to regularly test the performance of models in BioNeMo
and prevent changes to codebase that cause regression of their performance.  Currently, we run automatically tests in JET for

#### Automatic unit-tests in MR process to `dev`

 Tests are defined in [bionemo/training-unit-test](https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/tree/bionemo/training-unit-tests/recipes) and can be triggered in each MR in BioNeMo.
 They consist of
 * short trainings covering pretraining, pretraining with downstream tasks and fine-tuning on downstream task
 * and tests of the inference

for `MegaMolBART`, `ESM1` and `Prott5` across different gpus, precisions and batch sizes **but for only 1 node** (!)

The tests are run on a docker build on the modified codebase from the MR.

MR cannot be approved if these tests are not passed by the modified codebase.


#### Automatic unit-tests for every commit to `dev`

   The same as tests as above but run on a updated `dev:latest-devel`. Their results contribute to
   the algorithms that generate baseline for each metric that is logged during the JET tests

#### Regular & scheduled functional-tests on `dev`
To be added shortly
