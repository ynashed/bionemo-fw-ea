import shutil

import pytest

from download_models import download_models, get_available_models, load_config, streamed_subprocess_call


@pytest.fixture(scope='module')
def download_models_for_test(tmpdir_factory):
    """Downloads common models between NGC and PBSS for testing."""
    config = load_config()
    ngc_models = get_available_models(config, "ngc")
    pbss_models = get_available_models(config, "pbss")
    # Only download models which are common to both PBSS and NGC, so we can test the diff.
    common_models = list(set(ngc_models).intersection(set(pbss_models)))
    ngc_download_path = str(tmpdir_factory.mktemp("ngc_download"))
    pbss_download_path = str(tmpdir_factory.mktemp("pbss_download"))
    print(f"{common_models=}")
    print(f"{ngc_download_path=}")
    print(f"{pbss_download_path=}")
    # Download models from NGC
    download_models(config, common_models, "ngc", ngc_download_path)
    # Download models from PBSS
    download_models(config, common_models, "pbss", pbss_download_path)

    yield ngc_download_path, pbss_download_path

    # Teardown
    shutil.rmtree(ngc_download_path)
    shutil.rmtree(pbss_download_path)


@pytest.mark.internal
@pytest.mark.integration_test
def test_ngc_pbss_data_sync(download_models_for_test):
    """Tests that the models we have in PBSS are identical to those on NGC."""
    # Check the diff. This will raise an error if the directories differ.
    ngc_download_path, pbss_download_path = download_models_for_test
    stdout, stderr, retcode = streamed_subprocess_call(f"diff {ngc_download_path} {pbss_download_path}")
    if retcode != 0:
        raise ValueError(f"Models between NGC and PBSS differ! {stdout=}, {stderr=}")
