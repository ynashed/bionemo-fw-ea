# The Protein Data Bank ([PDB](https://www.rcsb.org/))
The Protein Data Bank is a public repository for experimental and computational chemists to publish results of protein-protein, protein-ligand, and isolated compound structure results. The database boasts over 213,221 experimentally derived structures, and 1,068,577 Computed Structure Models (CSM). Various API and toolings have been created to directly download from the server in [batch processes](https://www.rcsb.org/docs/programmatic-access/file-download-services), and some advanced commands for extracting specific groups of PDB IDs. 

Given its size, diversity in chemistry, and experimental validity, pruning techniques are often used to select what are seen as the optimal samples for the model problem at hand. Derived from the original PDB depositions, separate databases have been curated and maintained with supplemental information, or fully extracted for future academic use. 

## PDBBind {cite:p}`liu2017score`
It was the first database that systematically annotates the protein-ligand complexes in the Protein Data Bank (PDB) with experimental binding data. This database has been updated annually since its first public release in 2004. Data sets provided by PDBbind have been applied to many computational and statistical studies on protein-ligand interaction and various subjects. Total entries have exceeded ~20k for protein-protein, and protein-ligand interactions. 

DiffDock makes use of a Time-Split method for training, where data from the PDBBind database up to 2017 is used as training/validation, with 2018 through 2020 as the test set. The original PDBBind database can be found [here](http://pdbbind.org.cn/index.php), and an academic version that only includes structural information [here](https://zenodo.org/records/6408497)

## The Docking Benchmark 5.5 (DB5.5) {cite:p}`vreven2015updates`
The Docking Benchmark 5.5 (DB5.5) dataset updates version 5.0 with new high quality structures. The benchmarks consist of non-redundant, high-quality structures of protein-protein complexes along with the unbound structures of their components. Fifty-five new complexes were added to the docking benchmark, 35 of which have experimentally measured binding affinities. These updated docking and affinity benchmarks now contain 230 and 179 entries, respectively.

## Database of Interacting Protein Structures (DIPS) {cite:p}`townshend2019end`
As the PDB contains data of varying quality, DIPS aims to only include complexes that meet the
following criteria: ≥ 500 Å$^{2}$ buried surface area, solved using X-ray crystallography or cryo-electron microscopy at better than 3.5 Å resolution, only contains protein chains longer than 50 amino acids, and is the first model in a structure. As DB5 is also derived from the PDB, sequence-based pruning was used to ensure that there is no cross-contamination between train and test sets. The initial processing as well as the sequence-level exclusion yields a dataset of 42,826 binary complexes, over two orders of magnitude larger than DB5.
