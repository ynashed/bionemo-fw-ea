# BioNeMo2 Repo
To get started, please build the docker container using
```
bash bionemo2/launch.sh build
```

The repository holds the following structure
```

bionemo2/        # “new” development directory where all future bionemo
│                # development should occur, starting with ESM2-refactor
├─Dockerfile.bionemo2
├─setup.py
├─pyproject.toml
├─src/
│ └─bionemo/
│   ├─fw/                # Stable, reusable code goes here.
│   │ │                  # Code here must have arch diagrams.
│   │ ├─models/esm2      # Megatron “specs” might live here
│   │ └─data/            # Example location of data related tools
│   │   ├─scdl/          # Single cell dataloader example
│   │   └─fadl/          # Fasta dataloader example  
│   └─contrib/           # This is the “rapid path” location where rapid
│     │                  # external contributions can be merged
│     └─customer_model/  
│       └─...
├─tests/             # All tests live here. Test code must follow
│ ├─contest.py       # the same directory structure as the 
│ │                  # source code structure
│ ├─bionemo/
│ │ └─fw/
│ │   ├─data/
│ │   └─scdl/
│ │     └─test_scdl.py  # pytest example
│ └─contrib/
│   └─external_model/
│     └─test_external_model.py
├─scripts/           # Launchers of the framework, examples.
│ └─configs/         # Configs will live under scripts, not necessarily
│                    # in this exact directory structure.	
├─notebooks/         # Jupyter notebook examples. Solution architect
│                    # examples are expected to go here.
├─docs/              # Might inherit from notebooks
│ └─Dockerfile.docs  # Container to build documentation
│
│ # Below this line is all prior BioNeMo code from 
│ # https://gitlab-master.nvidia.com/clara-discovery/bionemo
├─bionemo/
├─README.md
├─CONTRIBUTING.md
├─.gitlab-ci.yaml
├─…
```
# TODO: Finish this.

## Models
### Geneformer
#### Get test data for geneformer
```bash
aws s3 cp \
  s3://general-purpose/cellxgene_2023-12-15_small \
  /workspace/bionemo/data/cellxgene_2023-12-15_small \
  --recursive \
  --endpoint-url https://pbss.s8k.io
```
#### Running
```bash
NVTE_FLASH_ATTN=0 python scripts/singlecell/geneformer/pretrain.py 
```