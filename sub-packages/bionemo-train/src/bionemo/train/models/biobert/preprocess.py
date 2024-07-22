from pathlib import Path

from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess


__all__ = ("make_geneformer_preprocessor",)


def make_geneformer_preprocessor(train_data_path: Path) -> GeneformerPreprocess:
    return GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
