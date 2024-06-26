# Running tests on BioNeMo's models using JET in CI
Tests initiated through JET (visit [JET Documentation](https://jet.nvidia.com/docs/tutorials/intro/)) in BioNeMo CI are trainings or testing/inference tasks related to BioNeMo's models which log performance, 
test if the scripts are executed correctly and optionally test for partial convergence. The SLURM jobs are executed on `draco-oci-iad`.


The configurations for tests launched in JET are located in `internal/jet/workloads/`:
* folder `training-inference-unit-tests` contains the training and inference smoke tests that are executed in merge request (MR) pipelines and pipelines triggered by commits to `dev` branch.
* folder `partial-conv-trainings` contains longer partial convergence tests that are scheduled nightly, weekly, or biweekly over the weekend.

### Configuration of JET tests
Each folder in `internal/jet/workloads/` consists of subfolders with configurations
* `builds` - defines how to get the docker image to execute scripts;
* `recipes` - contains configurations of trainings or prediction/testing tasks which passed to JET with scripts (python commands) to execute;

Remark that configurations in the folder `recipes` use python commands which
* runs on `draco-oci-iad`
* access model checkpoints and data from `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet` on draco-oci-iad
* launch training and inference scripts from `examples` 
* launch inference scripts from `bionemo/model/infer.py`

Therefore, any modifications to these files should also take into account the configurations in `internal/jet/workloads/training-inference-unit-tests/recipes` or `internal/jet/workloads/partial-conv-trainings/recipes`

The CI stages related to JET are named
* `jet-configure` - where the default docker image and additional attributes related to a pipeline are provided in the configurations in `builds` and `recipes`;
* `jet-trigger` - which triggers [JET CI](https://gitlab-master.nvidia.com/dl/jet/ci) using modified configs from `training-inference-unit-tests` or `partial-conv-trainings`. 
  * the tests are submitted to `draco-oci-iad` as slurm jobs
  * the stage waits for their completion and examines their execution
  * the logs are sent to a database
* `jet-test` - which collects outputs of the scripts executed in the previous stage using [JET API](https://gitlab-master.nvidia.com/dl/jet/api) 
  * summarizes their results collectively and per each test
  * prints docker info and scripts for debugging if tests failed

See [.gitlab-ci.yml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/.gitlab-ci.yml?ref_type=heads) for details.
## 1.  Standard sections of JET configurations located in the recipes folder
JET configs in the folders `recipes` have the following gsections shared among all use cases
* `labels` - key-value labels used for filtering and aggregating logs of JET jobs, they are updated in `jet-configure` stage in BioNeMo CI, see [.gitlab-ci.yml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/.gitlab-ci.yml?ref_type=heads#L737)
* `key_segments` - outlines the parameters that form the JET job key. Parameters from either the `spec` or `products` section can be added (e.g., `max_steps: max_steps`), removed (e.g., `extra_overwrites: False`), or renamed (e.g., `val_check_interval: val_check`) within the job key string. Parameters  `model`, `variant`, `nodes`, `gpus` and `precision` are by default used to form the JET job key
* `launchers` - specifies the configuration of available launchers along with the mounts mapping that can be utilized to execute the script defined in the JET configuration (workload).See [base_config.yaml](https://gitlab-master.nvidia.com/dl/jet/ci/-/blob/bionemo/base_config.yaml?ref_type=heads) from JET CI for a list of launchers enabled for BioNeMo and [JET Documentation](https://jet.nvidia.com/docs/concepts/jet-workloads/launchers/) for more information.
* `spec` - defines a JET workload including the variables and script to be executed (`script`), the docker image to be used (`build`) and the time limit for the SLURM job (`time_limit`, in sec), see [JET Documentation](https://jet.nvidia.com/docs/faq/#what-special-variables-can-i-use-in-my-scripts)
* `metrics` - specifies metric keys to track and display in high level logs, not relevant for BioNeMo
* `products` - specifies a grid of parameters to run `spec.script` section against
* `tests_spec` - defines a testing mechanism for JET job

### 1.1 Details about the `products` section
Imagine wanting to test a metric that varies based on the parameters of the same script, or simply wanting to test the execution of scripts with different input parameters.
You can associate the construction of the parameter grid with the workload `products` section. 

The command in the `spec.script` subsection varies across a grid of parameters which are defined by:
* number of `nodes` (e.g. `1`)
* variables specifying command to executed defined by `domain`, `model`, `variant` and `config_name`. 
  * `config_name` should be a test config with the lightest training specific for unit testing using sample data
* number of `gpus` (e.g., `1`, `8`)
* `precision` (e.g., `16`, `bf16`, `32`)
* batch sizes ( e.g., `32`, `64`)
* other model-specific parameters of interest
The first five parameter groups are mandatory in any JET config file.

These parameters are supplied to the python command. The number of tested cases per one JET config is determined by the size of the parameter grid.
You can find the related section in JET config similar to
```bash
products:
  - nodes: [1]
    gpus: [8]
    precision: [32]
    domain: [ protein ]
    batch_size: [ 1 ]
    model: [ openfold ]
    variant: [ train ]
    config_name: [openfold_initial_training_test]
    kalignu: [False, True]
```
which form `1**8 * 2` number of parameter pairs to test

### 1.2 Details about the `tests_spec` section
The `tests_spec` section is is related to the JET testing mechanism, see [JET Documentation](https://jet.nvidia.com/docs/tutorials/jet-tests/jet-tests-in-jet-scripts/#tests-associated-with-the-workload-product). The section outlines a set of rules to apply to the job's logs or final metrics obtained during the trainings. If the tests fail, the BIoNeMo CI stage jet-trigger does not fail; however, the jet-test stage does and provides details about the failure.
For instance, the testing logic to verify if training scripts are executed correctly is the following
```bash
tests_spec:
  - logic_type: static
    logic_spec:
      exit_codes:
        - 0
```

## 1. JET smoke tests
JET pipeline with smoke tests is included in
* merge request pipeline (disable it by applying the MR label `JET_NOT_REQUIRED`)
* is executed as part of the mandatory CI pipeline for each commit to the `dev` branch

Additionally, it can be executed as a web-triggered pipeline for any branch. Refer to section 3.1 below. 
Set `PYTEST=false` flag to disable the test stage (pytest) if not needed to get results faster.

The test configurations can be found under `internal/jet/workloads/training-inference-unit-tests`.
The tests should be structured as brief training or inference scripts, ideally lasting less than 5 minutes and using sample data. They should verify the execution of these scripts for any relevant parameters to ensure proper functionality after codebase changes.

The jobs in JET are executed on `draco-oci-iad` using BioNeMo's container. 
### 1.1 How is the sample data provided to the JET smoke test?
If a sample data for a training or inference config is not available in the docker container,
it must be placed under `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/data/test_data` on `draco-oci-iad`. It must be kept updated as well.
There is a section in JET config related to the specification of the mounted volumes
```bash
launchers:
  type:slurm:
    ntasks_per_node: '{gpus}'
  name:dgxa100_dracooci:
    mounts:
      /workspace/bionemo/examples/tests/test_data/openfold_data: /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/data/test_data/openfold_data
```
See [internal/jet/workloads/training-inference-unit-tests/recipes/config_openfold.yaml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/internal/jet/workloads/training-inference-unit-tests/recipes/config_openfold.yaml?ref_type=heads) for an example.

### 1.2 How are the model checkpoints provided to the JET smoke test?
Model checkpoints must be placed under `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/models` on `draco-oci-iad`. They must be kept updated as well.
```bash
launchers:
  type:slurm:
    ntasks_per_node: '{gpus}'
  name:dgxa100_dracooci:
    mounts:
      /workspace/bionemo/models: /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/models
```
See [internal/jet/workloads/training-inference-unit-tests/recipes/config_openfold.yaml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/internal/jet/workloads/training-inference-unit-tests/recipes/config_openfold.yaml?ref_type=heads) for an example.
### 1.3 Adding a new case to JET smoke tests

The steps to add a new smoke test configuration to JET are as follows:
1. Refer to the configurations in other config files under `internal/jet/workloads/training-inference-unit-tests` 
2. Specify the command to run by setting the `domain`, `model`, `variant`, and `config_name` variables.
3. Specify the mandatory test cases: `nodes=1`, `gpu`, `precisions`, `batch_size`
4. Specify other model-specific parameters required for executing the command during testing
5. If the sample data is not available in a bionemo docker container, place it in the following location `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/data/test_data` on `draco-oci-iad`
6. If any new model checkpoints are required, place them under `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/models` on `draco-oci-iad`

### 1.4 How to test JET configs under development
To test JET configurations, run the JET-only pipeline either for a merge request or using the web-based pipeline creator as described in section 3.

## 2. JET convergence tests
JET convergence tests involve longer training sessions for models which are orchestrated by JET. 
The parameterization of these trainings is similar to JET smoke tests, with the key difference being that they use full data to test production-grade training configurations. 
The convergence tests are designed to identify regressions in training accuracy. To do so, they test expected properties of the tested training sessions.
See configs in `internal/jet/workloads/partial-conv-trainings`.

The convergence tests launches `2.5-4h trainings` using many nodes hence they are run nightly as a [scheduled pipeline in BioNeMo repository](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/pipeline_schedules).
You can trigger JET convergence tests from the web-based pipeline creator as explained in section 3.1.2.

### 2.1 How is the full data provided to JET convergence tests?
The full data needed for a training must be placed under `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/data/` on `draco-oci-iad`. It must be kept updated as well.
In JET config there is a section related to the specification of the mounted volumes, ie
```bash
launchers:
  type:slurm:
    ntasks_per_node: '{gpus}'
  name:dgxa100_dracooci:
    mounts:
      /workspace/bionemo/data: /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/jet/data
```
### 2.2 How are the model checkpoints provided to the JET convergence test?
See section 1.2

### 2.3 How is the regression tested by JET convergence tests?
Additionally to examining an execution of the python script, JET convergence tests pipeline verifies that the final training metrics fall within the expected ranges (baselines), the tests include the following section
```bash
tests_spec:
  - logic_type: static
    logic_spec:
      exit_codes:
        - 0
      baselines:
        val_lddt_ca:
          operator: range
          max: 13.0
          min: 11.0
```
Visit [JET Documentation](https://jet.nvidia.com/docs/tutorials/jet-tests/jet-tests-in-jet-scripts/) for more details.
### 2.4 Dashboard for regression tracking & reporting
The tests results are displayed, summarised and tracked in the [dashboard](http://clara-discovery.gitlab-master-pages.nvidia.com/dashboards) deployed by [BioNeMo's dashboard repository](https://gitlab-master.nvidia.com/clara-discovery/dashboards).

Every convergence test pipeline in BioNeMo CI ends with the automatic dashboard update regardless of the tests results. 
The stage `dashboard-update-trigger` triggers deploy pipeline in the dashboard repository. 
See [.gitlab-ci.yml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/.gitlab-ci.yml?ref_type=heads) for details.

## 3. How to trigger JET testing pipeline
JET pipeline can be triggered using BioNeMo CI. This section outlines the most common approaches. 
### 3.1 Run the JET pipeline using the web-based pipeline creator.
Running JET test in BioNeMo CI using web pipeline creator.
Visit [pipelines tab in BioNeMo repository](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/pipelines), click on [Run pipeline](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/pipelines/new) button 
and select branch name (ie `dev`)

#### 3.1.1 Running JET only pipeline with JET smoke tests
Set `PYTEST=false` and `JET=true` to run smoke tests in JET for your branch defined in configs located in `training-inference-unit-tests`

#### 3.1.2 Running JET only pipeline with convergence tests
Set `JET_CONV_TEST=true` to run partial convergence tests in JET defined in configs located in `partial-conv-trainings`

### 3.2 Running JET tests in a merge request pipeline

#### 3.2.1 Running JET only pipeline with smoke tests
Set label `PYTEST_NOT_REQUIRED` to run only smoke tests in JET for your MR defined in configs located in `training-inference-unit-tests`
#### 3.2.2 Disabling JET stage with smoke tests from a merge request pipeline
Set label `JET_NOT_REQUIRED` to disable the `jet` stage from a merge request pipeline

## 4. How to configure a model-specific JET Pipeline
You can run the JET pipeline for a specific model instead of all models. The advantage of this approach is a shorter waiting time.
### 4.1 Model-specific merge request pipeline
When creating a merge request, specify **ONE** model label related to the model name from the available labels under see [BioNeMo MR's labels list](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/labels), ie `OpenFold, ESM1, ESM2, MolMIM` 
If multiple labels are specified, only the first one will be used. Do not set any model-specific label to run a pipeline that tests all models.

### 4.2 Model-specific web triggered pipeline 
To test only one model with JET, select the model name from the available options for `JET_MODEL_NAME`. If no specific model is selected, the default value is all, and all available models will be tested in the JET pipeline.

## 5. Adding a new model to JET tests
If new model needs to be added to JET tests, make sure that
1. a new merge request label `NEW_MODEL_LABEL` is created corresponding to a new model in [BioNeMo's GitLab repository label section](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/labels)
2. add a new model name `NEW_MODEL_NAME` to available options for `JET_MODEL_NAME` in [.gitlab-ci.yml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/.gitlab-ci.yml?ref_type=heads)
3. add additional rule to the section `rules` in `jet-trigger` [.gitlab-ci.yml](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/dev/.gitlab-ci.yml?ref_type=heads) to enable runnig pipelines exclusively for the new model
```bash
- if: $CI_MERGE_REQUEST_LABELS =~ /NEW_MODEL_LABEL/ || $JET_MODEL_NAME == "NEW_MODEL_NAME"
  variables:
    JET_WORKLOADS_FILTER: type == 'recipe' and spec.model == 'new_model_name'
```
where `new_model_name` is the string under `model` in the section `products` in the new model specific JET tests configs 
```bash
products:
  - domain: [....]
    variant: [...]
    model: [new_model_name]
```


## 6. Getting information about the tests
Use method `internal/jet/get_results_from_jet.py` to query the results of BioNeMo's tests in JET and get information about failures, errors or duration.
The results can be provided for all tests or filtered by specific reference IDs, pipeline IDs, workload IDs, or a period of time. 
They can be saved to a `.json file`, returned as `pandas.DataFrame` or printed to the console (default).
### 6.1 Installing JET API
JET API is command-line interface for querying JET logs. One need to install JET API (preferably in a seperate conda env)
in order to use `internal/jet/get_results_from_jet.py`. After installation, JET API credentials need to be set 
by directly calling `jet secrets login;jet secrets pull` from the command line.

### 6.2 Useful commands
Please refer to the documentation for comprehensive information on `internal/jet/get_results_from_jet.py` and its parameterization.
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

* By default, the method filters out jobs which has been rerun ie due to infrastructure error. In order to get information about all jobs (including failed and rerun ones), append `--all`
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
One may want to save the query results, by appending `--save_dir <FOLDER_PATH>` to save details of the run tests to a `.json` file with predefined path

```bash
python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>
```