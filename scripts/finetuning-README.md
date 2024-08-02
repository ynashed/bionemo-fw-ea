# Finetuning

```bash
python \
    scripts/singlecell/geneformer/finetune.py \
    --data-dir test_data/cellxgene_2023-12-15_small/processed_data \
    --result-dir ./results \
    --experiment-name test_experiment \
    --num-gpus 1 \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --nemo1-init-path models/geneformer-10M-240530-step-115430-wandb-4ij9ghox.nemo
```

## Nemo files!!!

```
ckpt in:
/workspaces/bionemo-github/results/test_experiment/2024-07-31_23-01-09/checkpoints/test_experiment--reduced_train_loss=8.2760-epoch=0
```

## Non-nemo files!

```
ckpt in:
/workspaces/bionemo-github/results/test_experiment/2024-08-01_22-48-48/checkpoints/test_experiment--reduced_train_loss=9.9996-epoch=0
``

## Todo
* look at: https://github.com/NVIDIA/NeMo/blob/r2.0.0rc1/tests/lightning/test_dist_ckpt.py#L83
