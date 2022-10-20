# Dataset Creation

Creating Train/Validation/Test dataset objects is a core step in most deep
learning training procedures.

For BioNeMo, we have provided a utility for creating these datasets, in
`nemo_chem.data.utils.build_train_valid_test_datasets`.

Assuming an `OmegaConf` object `conf`, and `pytorch_lightning.Trainer` object
`trainer` have already been created, you can use the following to create a
train/val/test split:

```python
from nemo_chem.data.utils import build_train_valid_test_datasets
number_of_samples = [
    10000, # train
    2000, # validation
    2000, # test
]
train_dataset, val_dataset, test_dataset = build_train_valid_test_datasets(
    cfg=conf,
    trainer=trainer,
    train_valid_test_num_samples=number_of_samples,
)
```

Note that this utility will create `MoleculesCSVDataset` instances by default.

## Defining your own dataset creation methods.

In order to use `build_train_valid_test_datasets` with a custom
dataset, an argument `dataset_factory` lets the user add their method for
creating datasets. This allows for arbitrary options, such as allowing multiple
data types for the same model, or even enabling this train-validation-test split
across very disparate data-types.

The `dataset_factory` should be of the `DatasetFactorySpec` type. Broadly,
this base class requires one method: a `create_dataset` method
that takes a dictionary `options` as it's only argument. The method can assume
`options` has the following keys:

    * `'cfg'`: an `OmegaConf` configuration object,
    * `'num_samples'`: the number of samples to use for the dataset (if applicable),
    * `'name'`: the name of the split (i.e., train/val/test).
    * `'dataset'`: An identifier for the dataset's location, most often the filepath(s).

Example of defining and using a new `DatasetFactorySpec`:

```python
from nemo_chem.data.utils import (
    build_train_valid_test_datasets,
    DatasetFactorySpec,
    get_filepath,
    cfg_get_key
)
from some_fancy_module import CustomPytorchDataset

class CustomDatasetFactory(DatasetFactorySpec):

    def create_dataset(self, options):
        filepath = get_filepath(options)
        is_random = cfg_get_key(options['cfg'], 'is_random', False)
        return CustomPytorchDataset(filepath=filepath, is_random=is_random)

```
