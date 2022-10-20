import pytest

from bionemo.utils.remote import GRCh38p13_ResourcePreparer, Hg38chromResourcePreparer
from bionemo.utils.preprocessors import FastaSplitNsPreprocessor

# TODO: integration tests, also notice that this is smoke tests.
#
# What should be the check here, and what takes it too far?? 
#    - that all the files are properly chunked?
#    - that the files have the sizes we expect?
#    - that the files are free of Ns?
#    - that Hg38 dumps into a chroms directory?


# NOTE Right now these tests are not invoked automatically due to the name of the file, in the future we may want another way to prevent running by default.
@pytest.mark.slow
def test_GRCh38p13_ResourcePreparer():
    prep = GRCh38p13_ResourcePreparer()
    resources = prep.prepare()
    prepared_files = list(FastaSplitNsPreprocessor(fasta_filenames=resources).map())

@pytest.mark.slow
def test_Hg38chromResourcePreparer():
    preparer = Hg38chromResourcePreparer()
    resources = preparer.prepare()
    prepared_files = list(FastaSplitNsPreprocessor(fasta_filenames=resources).map())
