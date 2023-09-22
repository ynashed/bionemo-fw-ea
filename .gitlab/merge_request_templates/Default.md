# What does this MR do?

Add a one-line overview of what this PR aims to accomplish.

**Collection**: [Note which collection this MR will affect. e.g., molecule, protein, or core (for generic contributions)]

# Usage
* You can potentially add a usage example below

```python
# Add a code snippet demonstrating how to use this 
```

# Before your PR is "Ready for review?"
**Pre checks**:
- [ ] Did you make sure that all tests pass? (run `pytest` in root of bionemo)
- [ ] Did you make sure that training and inference-related scripts pass? (run `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training` in the root of bionemo, only applicable if you have implemented changes to training and inference-related objects. See `CONTRIBUTING.md` for more details.
- [ ] Did you set `JET_NOT_REQUIRED` label as one of this MR's labels if your MR is eligible for NOT running `jet` stage (and tests in JET)? See `CONTRIBUTING.md` for information about MR with the type of changes to the codebase that can be merged without `jet` stage being triggered.
- [ ] Did you write any new necessary unit tests of added utilities (added to `tests`) or of training and inference for a new model (in `examples/tests`)? See `CONTRIBUTING.md` for more details.
- [ ] Did you add or update any necessary documentation (including comments in config files for all new config values)?
  

# When to label your MR as `JET_NOT_REQUIRED`
Most of the changes to files with extensions different than `*.py`, `*.yaml`, `*.yml`,  `Dockerfile` or `requirements.txt` DO REQUIRE `jet` stage to be run, should be carefully tested and ARE NOT eligible to use `JET_NOT_REQUIRE` label as they affect model performance or convergence. 
The changes to the codebase that are eligible for using `JET_NOT_REQUIRE` label are:
* docstrings update in `.py` files
* code cleanup not related to refactoring of code (ie deleting unused imports or blank lines, improving lines formatting) in `*.py` files
* improving hydra configs docstrings (comments and descriptions) in  `*.yaml`, `*.yml`
* changes to `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence. Changes that **REQUIRE** `jet` stage are, for instance, python package update or a NeMo container version update.
* updating files with extensions different than `*.py`, `*.yaml`, `*.yml`,  `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence


