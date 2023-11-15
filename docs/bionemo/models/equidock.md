# EquiDock

## Model Overview

EquiDock {cite:p}`ganea2021independent` is an SE(3)-equivariant model that can predict protein-protein complex formation from two invidual proteins. EquiDock treats both the proteins as rigid bodies and assumes no conformation change during the formation of the complex.

## Intended Use

Almost all biological processes involve proteins interacting with each other. Prediction of protein complexes can be used understand disease progression and also in the design of novel proteins that can preferentially interact with other proteins of choice.

## Model Architecture

EquiDock was implemented using the BioNeMo framework. EquiDock uses Independent E(3)-Equivariant Graph Matching Network (IEGMN) that performs graph feature embedding updates in addition to E(3)-equivariant coordinate updates. EquiDock uses a novel attention-based selection of keypoints to represent binding pocket of each protein via a differentiable variant of the Kabsch algortihm {cite:p}`kabsch1976solution`. In addition, the EquiDock architecture also gaurantees commutativity ie., docking protein 1 with respect to protein 2 gives identical results as docking protein 2 with respect to protein 1.

## Limitations

EquiDock is designed to treat proteins as rigid bodies and would not work well with proteins that undergo significant change in conformation upon binding.

## Training

### Dataset

The EquiDock model was trained on two datasets: The Docking Benchmark 5.5 (DB5.5) dataset {cite:p}`vreven2015updates` consisting of high quality structures and the Database of Interacting Protein Structures (DIPS) {cite:p}`townshend2019end` mined from the Protein Data Bank.

### Infrastructure and Configuration

The EquiDock model can be trained from scratch with the code and datasets provided. The `equidock_dips.nemo` and `equidock_db5.nemo` model checkpoints provided here were adapted from the public chekpoints available on [GitHub](https://github.com/octavian-ganea/equidock_public/tree/main/checkpts).

## License

EquiDock is provided under the {{model_license_slug}}.