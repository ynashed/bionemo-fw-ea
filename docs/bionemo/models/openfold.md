# OpenFold
#  Model Overview

## Description:
This is an OpenFold implementation under BioNeMo framework, derived from public OpenFold and DeepMind AlphaFold-2. This checkpoint was fine-tuned on the initial training checkpoint from the public OpenFold team.

OpenFold predicts protein structures from protein sequence inputs and optional multiple sequence alignments (MSAs) and template(s). This implementation supports initial training, fine-tuning and inference under BioNeMo framework. Detailed examples can be found under `examples/protein/openfold` within BioNeMo framework repository.

Users are advised to read the licensing terms under [public OpenFold](https://github.com/aqlaboratory/openfold) and [DeepMind AlphaFold-2](https://github.com/google-deepmind/alphafold) repositories as well as our copyright text.

This model is ready for commercial use. <br>

## References:
To cite OpenFold:
```bibtex
@article {Ahdritz2022.11.20.517210,
   author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
   title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
   elocation-id = {2022.11.20.517210},
   year = {2022},
   doi = {10.1101/2022.11.20.517210},
   publisher = {Cold Spring Harbor Laboratory},
   URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
   eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
   journal = {bioRxiv}
}
```

To cite AlphaFold-2:
```bibtex
@Article{AlphaFold2021,
 author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
 journal = {Nature},
 title   = {Highly accurate protein structure prediction with {AlphaFold}},
 year    = {2021},
 volume  = {596},
 number  = {7873},
 pages   = {583--589},
 doi     = {10.1038/s41586-021-03819-2}
}
```

If you use OpenProteinSet in initial training and fine-tuning, please also cite:
```bibtex
@misc{ahdritz2023openproteinset,
     title={{O}pen{P}rotein{S}et: {T}raining data for structural biology at scale},
     author={Gustaf Ahdritz and Nazim Bouatta and Sachin Kadyan and Lukas Jarosch and Daniel Berenberg and Ian Fisk and Andrew M. Watkins and Stephen Ra and Richard Bonneau and Mohammed AlQuraishi},
     year={2023},
     eprint={2308.05326},
     archivePrefix={arXiv},
     primaryClass={q-bio.BM}
}
```

## Model Architecture:
**Architecture Type:** Pose Estimation  <br>
**Network Architecture:** AlphaFold-2 <br>

## Input:
**Input Type(s):** Protein Sequence, (optional) Multiple Sequence Alignment(s) and (optional) Strutural Template(s) <br>
**Input Format(s):** None, a3m (text file), hhr (text file) <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** None <br>

## Output:
**Output Type(s):** Protrin Structure Pose(s), (optional) Confidence Metrics, (optional) Embeddings <br>
**Output Format:** PDB (text file), Pickle file, Pickle file <br>
**Output Parameters:** 3D <br>
**Other Properties Related to Output:** Pose (num_atm_ x 3), (optional) Confidence Metric: pLDDT (num_res_) and PAE (num_res_ x num_res_), (optional) Embeddings (num_res_ x emb_dims, or num_res_ x num_res_ x emb_dims) <br>

## Software Integration:
**Runtime Engine(s):**
* NeMo, BioNeMo <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux]

## Model Version(s):
OpenFold under BioNeMo framework  <br>

# Training & Evaluation:

## Training Dataset:
**Link:**  [PDB-mmCIF dataset](https://www.rcsb.org), [OpenProteinSet](https://arxiv.org/abs/2308.05326)  <br>
**Data Collection Method by dataset** <br>
* PDB-mmCIF dataset: [Automatic] and [Human] <br>
* OpenProteinSet: [Automatic] <br>

**Labeling Method by dataset** <br>
* [Not Applicable] <br>

**Properties:** PDB-mmCIF dataset: 200k samples of experimental protein structures. OpenProteinSet: 269k samples on sequence alignments. <br>
**Dataset License(s):** PDB-mmCIF dataset: [CC0 1.0 Universal](https://www.rcsb.org/pages/usage-policy). OpenProteinSet: [CC BY 4.0](https://registry.opendata.aws/openfold/).

## Evaluation Dataset:
**Link:**  [PDB-mmCIF dataset](https://www.rcsb.org), [OpenProteinSet](https://arxiv.org/abs/2308.05326)  <br>
**Data Collection Method by dataset** <br>
* PDB-mmCIF dataset: [Automatic] and [Human] <br>
* OpenProteinSet: [Automatic] <br>

**Labeling Method by dataset** <br>
* [Not Applicable] <br>

**Properties:** PDB-mmCIF dataset: 200k samples of experimental protein structures. OpenProteinSet: 269k samples on sequence alignments. <br>
**Dataset License(s):** PDB-mmCIF dataset: [CC0 1.0 Universal](https://www.rcsb.org/pages/usage-policy). OpenProteinSet: [CC BY 4.0](https://registry.opendata.aws/openfold/).

## Inference:
**Engine:** NeMo, BioNeMo, Triton <br>
**Test Hardware:** <br>
* [Ampere] <br>
* [Hopper] <br>

## Benchmarks

### Accuracy benchmark

There are two stages of training OpenFold: initial-training and fine-tuning. 4 checkpoints are available for download: a pair of initial-training and fintuining available publicly and converted to .nemo format, and another pair of in-house trained checkpoints. All checkpoints are benchmarked against [CAMEO benchmark](https://cameo3d.org/modeling/3-months/) with proteins dated from 2021-09-17 to 2021-12-11. This validation set is available through training data.

Benchmark results (lDDT-cα) for checkpoints trained using BioNeMo framework:

|                                 | initial-training | fine-tuning* |
|---------------------------------|------------------|--------------|
| CAMEO  2021-09-17 to 2021-12-11 |       89.82      |      91.0    |

*this checkpoint was fine-tuned starting from **public** inital-training, and is available for download via NGC (using the `download_models.py` script).

### Training Performance Benchmarks

Training speed was tested 16 DGX-A100 (128 GPUs) with 80GB of memory, with a single protein (micro batch size of 1) per GPU.

|                        | initial-training | fine-tuning |
|------------------------|------------------|-------------|
| number of steps        |      80,000      |    12,000   |
| training step time (s) |       6.06       |    24.91    |

Note that in the default configuration of OpenFold training shipped in BioNeMo, there is validation every 200 steps, which takes about 3 minutes of duration. Initial training therefore takes approximately 6.5 days.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
