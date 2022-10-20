## Overview
Currently there are many core functionalities that are distributed and repeated
throughout several NeMo/BioNeMo models. This proposal addresses
centralize many of the core operations of BioNeMO models to
allow for a simplified workflow for future extensions of BioNeMo models.

## What is it:

A data module for BioNeMo is an interface that can be implemented, similar to
Pytorch Lightning's [`LightningDataModule`](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html),
that encapsulates all of the steps needed to setup data processing
for BioNeMo models, which is the primary component that varies between most
models. In BioNeMo, the data module can be used with a model to adapt the model
to the specific data use-case.


Example:

```python
from bionemo.core import BioNeMoDataModule
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model

class DNAT5DataModule(BioNeMoDataModule):
    # define dna specific data module behavior here
    def __init__(self, ...):
        ...

class BioNeMoT5Model(MegatronT5Model):
    def __init__(self, cfg, trainer, data_module, *args, **kwargs):
        self.data_module = data_module
        super().__init__(cfg, trainer, *args, **kwargs)

    def _build_tokenizer(self):
        self.tokenizer = self.data_module.build_tokenizer(self)

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)
        self.data_module.adjust_train_dataloader(self, self._train_dl)


dnat5_model = BioNeMoT5Model(cfg, trainer, DNAT5DataModule())
```

This approach isolates dependencies on NeMo to a single place per model and
allows the model logic to be implemented in a data-agnostic manner.
The encapsulation facilitates many general purpose solutions to common problems that
can be enabled for novel data use-cases simply by implementing the Data Module's
interface.


## Justification:
* Examples:
    * Different models in NeMo have different interfaces to accomplish the same tasks:
        * e.g., `build_train_valid_test_datasets` vs. `_build_train_valid_test_datasets` in `MegatronT5Model` vs. `MegatronBertModel`
        * With Data Modules, these dependencies on NeMo can be delegated to a single
          instance per kind of model.
        * Implementing a new biologically-driven task in BioNeMo can then
          be accomplished by only focusing on the data
    * Inference model classes have to be created on a per-model basis
        * Some models e.g., ProtT5nv, ESMnv, DNABERT are very similar and only vary in key aspects about handling the specific data/model, which are specifics that can be encapsulated.
        * The models should be able to perform most of these tasks without knowledge
          of what the data looks like.
    * Fine-tuning model
        * A fine-tuned language model inherited from ModelPT has to re-implement some functionality that is seemingly general, but is limited to MegatronBaseModel, e.g., `build_train_test_valid datasets` in general or tokenization for NLP models.
        * With the Data Module approach, this logic should only need to be implemented once in order
          to be re-used for various other applications, and the part that varies (data) can be
          customized in each instance.


## Details of the BioNeMo Data Module:

An example stub for the proposed base model could look like:
```python
# bionemo/core/datamodule.py
class BioNeMoDataModule(object):
    def __init__(self, cfg, trainer):
        pass

    @abstractmethod
    def train_dataset(self):
        pass

    @abstractmethod
    def val_dataset(self):
        pass

    @abstractmethod
    def test_dataset(self):
        pass

    def adjust_train_dataloader(self, model, dataloader):
        pass

    @abstractmethod
    def adjust_val_dataloader(self, model, dataloader):
        pass

    @abstractmethod
    def adjust_test_dataloader(self, model, dataloader):
        pass

    def init_num_samples(self):
        pass

    def sample_train_dataset(self, dataset):
        pass

    def sample_val_dataset(self, dataset):
        pass

    def sample_test_dataset(self, dataset):
        pass

    def get_global_batch_size(self):
        pass

    def get_max_train_steps(self):
        pass

    def get_total_eval_batches(self):
        pass

    def get_total_test_batches(self):
        pass

    def get_sampled_train_dataset(self):
        pass

    def get_sampled_val_dataset(self):
        pass

    def get_sampled_test_dataset(self):
        pass

    def get_train_dataset(self):
        pass

    def get_val_dataset(self):
        pass

    def get_test_dataset(self):
        pass


class BioNeMoNLPDataModule(BioNeMoDataModule):

    @abstractmethod
    def build_tokenizer(self, model):
        pass

```


This configuration will enable the following:
* Centralizes core operations and unify existing models/inference wrappers:
    * Plugin configuration
    * Functionality needed for inference
    * Any time we want to deploy new features across all models, changes only need to be applied in one place
* Standardizes many aspects of NeMo model that our users are likely to want to modify:
    * Dataset/data type creation
        * There are many representations our users could potentially use for DNA, proteins, small molecules, etc.
        * Among these, there are also plenty of data formats that might need to be accommodated, especially when fine-tuning tasks are considered. Examples:
            * CSV -> including SMILES or other structured data
            * FASTA
            * FASTQ
            * Tensors (e.g., for conformers, which may have a different format
                as well)
            * Graph edge/node lists
    * Tokenization
        * Certain data types may have representations that are unique to them
            * e.g., nucleotide tokenizer needs acgt or it will give non-sense,
                doesn't make sense to make accessible to natural language
        * Side note: we might not always need/want tokenization? e.g., graph
            models
    * Model kind (e.g., T5 vs. BERT vs. BART vs. graph models)
* Provides a single place to guide creation of BioNeMo models (a la pytorch
    lightning module)
    * A developer guide is straightforward to write/read if there is a
        single (few) model(s) to inherit, that enable a variety of behavior via
        extension.
