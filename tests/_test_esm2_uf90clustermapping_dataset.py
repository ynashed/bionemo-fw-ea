from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer


def test_uf90_resampling_dataset_integration():
    '''
    This test requires:

    cluster_mapping_tsv: /data/uniref2022_05/fake-uniref/mapping.tsv
    uf50_datapath: /data/uniref2022_05/fake-uniref/Fake50.fasta 
    uf90_datapath: /data/uniref2022_05/fake-uniref/Fake90.fasta 
    dataset_path: /data/uniref2022_05/fake_uf_map

    you can generate Fake50.fasta, Fake90.fasta, and mapping.tsv by running
    '''
    
    
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('conf/uf90-resampling.yaml')
    trainer = setup_trainer(cfg, callbacks=[])

    # Required for the dataloaders
    from bionemo.model.utils import initialize_distributed_parallel_state
    initialize_distributed_parallel_state(local_rank=0, tensor_model_parallel_size=1,pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=0)

    # Preprocess the testing dataset
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.uf50_datapath,
        uf90_datapath=cfg.model.data.uf90_datapath,
        cluster_mapping_tsv=cfg.model.data.cluster_mapping_tsv,
        uf50_output_dir=cfg.model.data.dataset_path,
        uf90_output_dir=cfg.model.data.uf90.uniref90_path,
        val_size=cfg.model.data.val_size,
        test_size=cfg.model.data.test_size,
        force=False
    )

    # Then test that it does what we think it should do.
    train_ds, val_ds, test_ds = esm1nv_model.ESM2nvModel._build_train_valid_test_datasets(trainer, cfg.model, keep_uf50=True)
    for ds in [train_ds, val_ds, test_ds]:
        for uf50, uf90 in zip(ds.uniref50_dataset, ds):
            # Compares the IDs, where we manually constructed them to be nearly-equal.
            assert uf50.split("UniRef50_")[1] in uf90['sequence_id'].split("UniRef90_")[1]