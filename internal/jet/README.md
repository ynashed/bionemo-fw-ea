# Running tests of BioNeMo's models using JET in CI

Tests of launched via JET in CI are short trainings or testing/inference tasks of BioNeMo's models which log performance 
and partial convergence during script execution.

The script launched in JET are located in `internal/jet/workloads/training-inference-unit-tests`. The folder consists of subfolders with configurations
* `builds` - defines how to get the docker image to execute scripts;
* `recipes` - contains configurations of trainings or prediction/testing tasks which passed to JET with scripts (python commands) to execute;

Remark that configurations in the folder `recipes` use python commands which 
* download model checkpoints via `download_models.py`
* download sample data if needed (model-specific)
* launch training scripts from `examples` 
* launch inference scripts from `bionemo/model/infer.py`
Hence any modifications to these files should also consider configs in `internal/jet/workloads/training-inference-unit-tests/recipes`

The stages in CI related to JET are named 
* `jet-configure` - where the default docker image and additional attributed of the logs related to a pipline are supplied to configs in `builds` and `recipes`;
* `jet-trigger` - which triggers [JET CI](https://gitlab-master.nvidia.com/dl/jet/ci) using modified configs in `internal/jet/workloads/training-inference-unit-tests` and waits for their completion. JET CI launches training scripts on `draco-oci`;
* `jet-test` - which collects outputs of the scripts executed in the previous stage using [JET API](https://gitlab-master.nvidia.com/dl/jet/api) and tests if the executed trainings and prediction tasks satisfy imposed condition, so far, if they were executed successfully; 

The tests in JET are short training or testing commands executed in a BioNeMo's docker container across a grid of parameters
such as different
* number of nodes: 1, 4
* number of gpus: 1, 8
* precisions: 16, bf16, 32
* batch sizes: two model-specific batch sizes ie 32, 64
* model parallel setting
which are supplied to the python command. Therefore, by default, each python command for training or testing is run across `3 * 2^3` combinations of these parameters that results in the same number of tested cases of a python training command.

### Running test for the first time
In order to run tests for the first time, JET API credentials need to be set. There are two ways how it can be set
* by appending flag `--setup_jet_api` when `python internal/jet/run_jet_downstream_pipeline.py ...` is executed for the first time
* or by directly calling `jet secrets login;jet secrets pull` from the command line.

### Getting information about the tests
Use method `internal/jet/get_results_from_jet.py` to query the results of BioNeMo's tests in JET and output some execution stats such as duration.
The results can be supplied for all BioNeMo's tests, only for specific reference or pipeline or workload ids, period of time
They can be saved to a .csv file or printed to the console (default)

* Get results for all tests run for BioNeMo account (commits to brunches or merge request pipelines), ie
```bash
python internal/jet/get_results_from_jet.py
```
* Get pipeline id-related results
```bash
python internal/jet/get_results_from_jet.py --pipeline_id 9469458
```

* Filter our jobs which has been rerun by attaching --most_recent
```bash
python internal/jet/get_results_from_jet.py ..... --most_recent
```

* Specifies period in the past to include jobs from up to now. The accepted formats are either dates from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00','2023-01-01') or durations, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
```bash
python internal/jet/get_results_from_jet.py ..... --duration 2w
```


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
