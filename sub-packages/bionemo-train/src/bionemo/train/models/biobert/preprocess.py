from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess


__all__ = ("make_geneformer_preprocessor",)


def make_geneformer_preprocessor(pretrain_config) -> GeneformerPreprocess:
    train_data_path = pretrain_config.train_data_path
    return GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )


def make_singlecell_datamodule(pretrain_config, preprocessed_resources):
    data = SingleCellDataModule(
        seq_length=pretrain_config.seq_length,
        tokenizer=preprocessed_resources["tokenizer"],
        train_dataset_path=pretrain_config.train_data_path,
        val_dataset_path=pretrain_config.val_data_path,
        test_dataset_path=pretrain_config.test_data_path,
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=preprocessed_resources["median_dict"],
        micro_batch_size=pretrain_config.micro_batch_size,
        global_batch_size=(
            pretrain_config.micro_batch_size
            * int(pretrain_config.num_nodes * pretrain_config.devices / pretrain_config.pipeline_model_parallel_size)
        ),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=pretrain_config.num_dataset_workers > 0,
        pin_memory=False,
        num_workers=pretrain_config.num_dataset_workers,
    )
    return data
