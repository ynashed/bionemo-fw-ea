from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer


def test_uf90_resampling_dataset_integration():
    # TODO(testing): this is functional/integration test between NeMo megatron and our Dataset class.
    # TODO: bring synthetic data generation script into tests somewhere.
    # TODO: locate the datasets required for this somewhere in the tests directory
    # TODO: update the config file to use the above paths.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('conf/uf90-resampling.yaml')
    trainer = setup_trainer(cfg, callbacks=[])

    # Required for the dataloaders
    from bionemo.model.utils import initialize_distributed_parallel_state
    initialize_distributed_parallel_state(local_rank=0, tensor_model_parallel_size=1,pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=0)
    train_ds, val_ds, test_ds = esm1nv_model.ESM2nvModel._build_train_valid_test_datasets(trainer, cfg.model)

    for ds in [train_ds, val_ds, test_ds]:
        for uf50, uf90 in zip(ds.uniref50_dataset, ds):
            # Compares the IDs, where we manually constructed them to be nearly-equal.
            assert uf50.split("UniRef50_")[1] in uf90['sequence_id'].split("UniRef90_")[1]
