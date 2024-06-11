# BioNeMo2 Repo
To get started, please build the docker container using
```bash
./launch.sh build
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
