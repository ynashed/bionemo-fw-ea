# Integration Tests

## End-to-end tests (`test_scripts_e2e.py`)

### General design

## Model pretraining tests (`test_model_pretrain_and_downstream.py`)

### General design

## Stop-and-go tests (`test_stop_and_go.py`)

### General design

Stop-and-go tests check against correct resuming from a saved checkpoint after interruption of training (e.g. due to a job timeout on a SLURM cluster). The single tests consists of three stages:
1. An initial training run that saves regular checkpoints
2. A termination signal that mimics the real termination signal stopping the training run
3. A resumed training run that takes the checkpoint and continues training after checking that a few conditions have been fulfilled

In the stop-and-go test, we have one main process running. This one starts a subprocess for the first training run that kills itself once a checkpoint itself via a callback. The main process than starts the second subprocess that picks up again from the checkpoint with a new callback that checks whether the important metadata agrees.

For the stop-and-go tests to work properly, there are two conditions that need to be fulfilled when it comes to timing of the callbacks:
1. The metadata that is saved and later used for comparison should be saved at the same time as the checkpoint from which training is resumed later on so that the numbers match.
2. The kill signal should occur after both the metadata and the corresponding checkpoint have been saved. A training run with 8 training steps was chosen to keep the test duration short, with validation and therefore checkpoint saving happening every 2 steps.

### Callback Setup 

The metadata saving and the killing of the process are two separate callbacks. The kill callback is now executed with the hook on_training_batch_start, executing at at the start of every training batch. To prevent it from killing the job at the start of training, we make the execution of the kill callback conditional on the presence of the metadata file: if there is no metadata file detected, the kill callback will do nothing. Once a metadata file is detected, the job is killed, meaning the job is killed as soon as the first training batch after the validation epoche starts, mimicking the desired behavior from option 1 closely.


### Code Locations

For this test to work, multiple files inside the BioNeMo codebase have to be created:
1. The test itself
2. The three new callbacks (MetaDataSave, KillAfterSignal and CheckpointIntegrityTest)
3. Code to integrate the callback setup into the trainer setup

Here more detail on each of the parts:

1. The test itself: placed in `examples/tests/test_stop_and_go.py`. Similar structure to test_scripts_e2e.py in the same directory (idea from Dorota Toczydlowska PL to leverage some of that code for the test), therefore sensible to put it in the same location.
2. The three new callbacks: placed in `bionemo/callbacks/testing_callbacks.py`. General folder for callbacks, and since there were no testing-related callbacks so far creating a separate file for it made sense.
3. Code to integrate callback setup into trainer setup: Added code to `bionemo/callbacks/utils.py` in the form of the more code to add_test_callbacks function. It already checked whether config arguments for blogger and PerfCallback are set, so integrated the two new callbacks in a similar way. These configs are then added/overridden in the test_stop_and_go.py script.

### How to add a stop-go-test for a new model

Stop-and-go tests can be extended in two main ways:

1. Add support for a new model.
2. Add suport for a new parameters to be checked between restarts.

#### Add support for a new model

For adding a new model, the `TEST_PARAMS` dictionary in `examples/tests/test_stop_and_go.py` should be extended with an entry for this new model. This entry should include 

- the path to the training script for this model (`script_path`)
- the key of the metadata (i.e. a parameter) that should be checked between restarts (`metadata_keys`). The valid arguments here are defined in `getter_function_map` in `bionemo/callbacks/testing_callbacks.py`.

#### Add support for a new parameters to be checked between restarts

For adding a new parameter, you have to add a getter function for this parameter in `bionemo/callbacks/testing_callbacks.py`. See and example of getter function such as `get_global_step` or `get_learning_rate`. This getter function should have the same function signature as the previous ones, i.e. it should accept the arguments `trainer: Trainer, pl_module: LightningModule`. This getter function will then be called upon both saving of the metadata and comparison of values after restart.
To add a new getter function, one needs to

1. add a keyword for this getter function in `getter_function_map`
2. map the keyword with the newly implement getter function

This is necessary to map from the keywords that are given in the `TEST_PARAMS` dictionary in `examples/tests/test_stop_and_go.py` to the actual getter functions. One cannot directly define and pass the getter functions in `examples/tests/test_stop_and_go.py` since the Hydra CLI does not permit proper passing of callables.