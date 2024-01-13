# Release Notes
## BioNeMo Framework v1.1
## New Models
* EquiDock for protein-protein docking pose prediction
* DiffDock for protein-ligand blind docking pose generation

### New Features
* Training recipes for EquiDock and DiffDock, including automated data processing and full configuration for training.
* Accelerated inference and training for DiffDock via fast tensor-product kernels.        
* Example tutorials for running inference using EquiDock and DiffDock.
* Recipes for running EquiDock and DiffDock on BCP and Slurm.
* Pipeline parallel supported for ESM-2nv.
* Migration of inference notebooks to using pytriton.

### Bug fixes and Improvements
* Faster pre-processing of data on BCP.
* Refactor of download_models.sh to download_models.py for easier CLI use.
* Refactor of install structure to move from /opt/nvidia to /workspace/bionemo. The environment variable $BIONEMO_HOME now points to the repo base and is required to be set for tests to pass.

## BioNeMo Framework v1.0
## New Models
* ESM-2nv for protein sequence representations, pretrained weights of ESM-2 650M and ESM-2 3B converted from HF checkpoint available.

### New Features
* Pre-training recipes for ESM-2nv, including automated data processing and full configuration for training
* Fine-tuning of ESM-2nv with encoder frozen or trainable
* Downstream task finetuning support for single-value classification (e.g. subcellular localization), single-value regression (e.g. meltome) and per-token classification (e.g. secondary structure) 
* Validation in loop to evaluate performance on downstream tasks during training
* Example tutorials for pre-training, fine tuning, and downstream tasks

## BioNeMo Framework v0.4.0
### New Models  
* ESM-1nv for protein sequence representations, pretrained weights available
* ProtT5nv for protein sequence representation and sequence-to-sequence tasks, pretrained weights available
### New Features
* Pre-training for all models, including automated data processing and full configuration for training
* Fine-tuning of MegaMolBART, ESM-1nv, and ProtT5nv with encoder frozen or trainable
* Downstream task example applications – secondary structure prediction for ESM-1nv and ProtT5nv, physchem prediction (lipophilicity, FreeSolv, ESOL) and retrosynthesis prediction for MegaMolBART
* Validation in loop to evaluate performance on downstream tasks during training: physchem prediction (MegaMolBART) and secondary structure prediction (ESM-1nv and ProtT5nv).
* Pipeline parallelism supported as a beta feature. Not fully tested.
* Example notebooks for pre-training, fine tuning, and downstream tasks

### Known Issues
* Data preprocessing on DGX Cloud is slow. Faster to do it on a local machine.
### New APIs
* BioNeMoDataModule - Encapsulates dataset instantiation in bionemo models so that many different datasets can be used with the same model
* EncoderFineTuning - Base class to facilitate implementation of downstream tasks built on embeddings from other models
