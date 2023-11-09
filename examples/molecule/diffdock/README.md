All configurations are controlled by the `conf/*.yaml` files.

To run training do `python train.py` and it will import the configurations from the `conf/train_base.yaml` file.
Note the data preprateion is automated by the `preprocess` module which itself loads configurations from the `conf/preprocess_data.yaml` file.

The `diffdock` code is split into three parts:

    1. the protein data converted into `PDBBind` object which is handled by the `bionemo.data.diffdock` module.

    2. the neural network models for score and confidence models that are at `bionemo.model.molecule.diffdock` module.

    3. the example files for driving training and inference that reside in `bionemo.examples.diffdock` module.