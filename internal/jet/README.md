# Running tests of BioNeMo's models using JET in CI
Tests launched via JET (visit [JET Documentation](https://jet.nvidia.com/docs/tutorials/intro/) ) in BioNeMo CI are trainings or testing/inference tasks of BioNeMo's models which log performance, 
test if scripts executed correctly and optionally test for partial convergence. The SLURM jobs are executed on `draco-oci-iad`

The script launched in JET are located in `internal/jet/workloads/`:
* folder `training-inference-unit-tests` corresponds to integration unit tests run in merge request (MR) pipelines and commit pipelines to `dev`
* folder `partial-conv-trainings` corresponds to longer partial convergence tests scheduled biweekly over weekend. 

### Configuration of JET tests
Each folder in `internal/jet/workloads/` consists of subfolders with configurations
* `builds` - defines how to get the docker image to execute scripts;
* `recipes` - contains configurations of trainings or prediction/testing tasks which passed to JET with scripts (python commands) to execute;

Remark that configurations in the folder `recipes` use python commands which 
* access model checkpoints and data from `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet`
* launch training scripts from `examples` 
* launch inference scripts from `bionemo/model/infer.py`
Hence any modifications to these files should also consider configs in `internal/jet/workloads/training-inference-unit-tests/recipes`

The stages in CI related to JET are named 
* `jet-configure` - where the default docker image and additional attributes of the logs related to a pipline are supplied to configs in `builds` and `recipes`;
* `jet-trigger` - which triggers [JET CI](https://gitlab-master.nvidia.com/dl/jet/ci) using modified configs from `training-inference-unit-tests` or `partial-conv-trainings` and waits for their completion. JET CI launches training scripts on `draco-oci`;
* `jet-test` - which collects outputs of the scripts executed in the previous stage using [JET API](https://gitlab-master.nvidia.com/dl/jet/api) and tests if the executed trainings and prediction tasks satisfy imposed condition, so far, if they were executed successfully; 

The jobs in JET are training or inference commands executed in a BioNeMo's docker container across a grid of parameters
such as different
* number of nodes: 1, 4
* number of gpus: 1, 8
* precisions: 16, bf16, 32
* batch sizes: two model-specific batch sizes ie 32, 64
* model parallel setting
which are supplied to the python command. Therefore, by default, each python command for training or testing is run across `3 * 2^3` combinations of these parameters that results in the same number of tested cases of a python training command.

## Running JET test in BioNeMo CI using web pipeline creator
Visit [Pipelines tab in BioNeMo repository](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/pipelines), click on [Run pipeline](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/pipelines/new) button 
and select branch name (ie `dev`)
### Specifying a model to run pipeline for
If only one model needs to be tested with JET, select model name from available model names for `JET_MODEL_NAME`.
If not selected otherwise, `all` is a default value and all available models will be tested in JET pipeline

### Running JET only pipeline with integration tests
Set `PYTEST=false` and `JET=true` to run short integration tests in JET for your branch defined in configs located in `training-inference-unit-tests`

### Running JET only pipeline with partial convergence tests
Set `JET_CONV_TEST=true` to run partial convergence tests in JET defined in configs located in `partial-conv-trainings`

## Running JET test in BioNeMo CI uin Merge Request
Control over JET pipeline in merge requests is done using label, see [BioNeMo MR's labels list](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/labels)
### Specifying a model to run pipeline for
Specify ONE model label related to model name from available labels, ie `OpenFold, ESM1, ESM2, MolMIM` when creating MR.
If many labels are specified, only first one will be used
### Running JET only pipeline with integration tests
Set label  `PYTEST_NOT_REQUIRED` to run only short integration tests in JET  for your MR defined in configs located in `training-inference-unit-tests`

## Getting information about the tests
Use method `internal/jet/get_results_from_jet.py` to query the results of BioNeMo's tests in JET and output some execution stats such as duration.
The results can be supplied for all BioNeMo's tests, only for specific reference or pipeline or workload ids, period of time
They can be saved to a .csv file or printed to the console (default)

#### Installing JET API
JET API is command-line interface for queries JET logs. One need to install JET API (preferably in a seperate conda env)
in order to use `internal/jet/get_results_from_jet.py`. After installation, JET API credentials need to be set 
by directly calling `jet secrets login;jet secrets pull` from the command line.

#### Useful commands
Please checkout docs of to get all information about `internal/jet/get_results_from_jet.py` and its parametrisation

* Get results for all tests run for BioNeMo account (commits to brunches or merge request pipelines), ie
```bash
python internal/jet/get_results_from_jet.py
```
* Get results for all tests run for BioNeMo account for specific branch, ie `dev`, ie
```bash
python internal/jet/get_results_from_jet.py --pipeline_type dev
```
* Get JET pipeline id-related results
```bash
python internal/jet/get_results_from_jet.py --pipeline_id 9469458
```

* By default the method filters out jobs which has been rerun ie due to infrastructure error. In order to get information about rerun jobs, including the rerun ones attach --all
```bash
python internal/jet/get_results_from_jet.py ..... --all
```

* Specifies period in the past to include jobs from up to now. The accepted formats are either dates from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00','2023-01-01') or durations, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
```bash
python internal/jet/get_results_from_jet.py ..... --duration 2w
```

* Get JET different level of verbosity by appending `-v` (duration info printed) `-vv` (job status and duration info printed) or `-vvv` (job status, detailed job info and duration info printed)
```bash
python internal/jet/get_results_from_jet.py --pipeline_id 9469458 -v
```


#### Format of the information about the tests
By default `internal/jet/get_results_from_jet.py` prints commands of the failed tests to the console and related docker info
to replicate the errors

The text printed do the command contains information about docker build such as
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
JET job logs: 
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
python download_artifacts.py --models all --model_dir ${MODEL_PATH} --verbose
python examples/molecule/megamolbart/pretrain.py --config-path $BIONEMO_HOME/examples/tests/conf/ --config-name megamolbart_test \
trainer.num_nodes=1 trainer.devices=8 trainer.precision=32 model.micro_batch_size=32 \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.create_wandb_logger=False ++exp_manager.create_tensorboard_logger=False ++exp_manager.exp_dir=/tmp/nemo_experiments/ ++exp_manager.create_checkpoint_callback=False ++exp_manager.resume_if_exists=False  hydra.searchpath=[file://$BIONEMO_HOME/examples/molecule/megamolbart/conf] \
++create_dllogger_callbacks=True ++create_trainer_metric_callback=True \
++dllogger_callbacks_kwargs.use_existing_dllogger=True ++dllogger_callbacks_kwargs.warmup=100 \
++dllogger_callbacks_kwargs.json_file=/jet/logs/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false/dllogger.json ++trainer_metric_callback_kwargs.log_path=/jet/assets/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false ++logs_dir=/jet/logs/recipe/megamolbart_pretrain_bionemo_perf-train_32_nodes-1_gpus-8_bs-32_dwnstrtask-false
```
One may want to save the
return_df
Or attach to your command `--save_dir <FOLDER_PATH>` to save details of the run tests to a .csv file with predefined path

`python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>`
