# Finetuning

```bash
python \
    scripts/singlecell/geneformer/pretrain.py \
    --data-dir test_data/cellxgene_2023-12-15_small/processed_data \
    --result-dir ./results \
    --experiment-name test_experiment \
    --num-gpus 1 \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 2048 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --nemo1-init-path models/geneformer-10M-240530-step-115430-wandb-4ij9ghox.nemo
```

## Todo
* look at: https://github.com/NVIDIA/NeMo/blob/r2.0.0rc1/tests/lightning/test_dist_ckpt.py#L83
