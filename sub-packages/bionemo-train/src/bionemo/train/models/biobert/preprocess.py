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


def make_geneformer_model_config(pretrain_config, precision):
    geneformer_config = BioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=pretrain_config.seq_length,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.relu,  # TODO(@jstjohn) check this
        qk_layernorm=True,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=False,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=True,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option=biobert_spec_option,
    )
    return geneformer_config
