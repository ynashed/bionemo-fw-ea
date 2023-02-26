import pytest
import os
import glob
from omegaconf import OmegaConf
import pathlib
from bionemo.data import Zinc15Preprocess
from bionemo.utils.tests import get_directory_hash

os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
ROOT_DIR = 'zinc15'
SAMPLE_DATA = os.path.join(os.environ['PROJECT_MOUNT'], 
                           'examples/molecule/megamolbart/dataset/ZINC-downloader-test.txt')
CONFIG = {'max_smiles_length': 512,
               'train_samples_per_file': 1000,
               'val_samples_per_file': 100,
               'test_samples_per_file': 100,
               'links_file': SAMPLE_DATA,
               'pool_size': 4,
               'seed': 0}
HEADER = 'zinc_id,smiles'
TRAIN_VAL_TEST_HASHES = {'train': '5d1b2856c12fa37972a68f12167eadf3', 
                         'val': '92f5a0e37644f849effd64e17412fb76', 
                         'test': '0d6332df6c3b985d00e0664e3d3fde07'}

##############

@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory, root_directory=ROOT_DIR):
    """Create tmp directory"""
    tmp_path_factory.mktemp(root_directory)
    return tmp_path_factory.getbasetemp()


@pytest.fixture()
def download_directory(tmp_directory):
    """Create the temporary directory for testing and the download directory"""
    download_directory = pathlib.Path(os.path.join(tmp_directory, 'raw'))
    download_directory.mkdir(parents=True, exist_ok=True)
    return download_directory


@pytest.fixture()
def output_directory(tmp_directory):
    """Create the directory for processed data"""
    download_dir = pathlib.Path(os.path.join(tmp_directory, 'processed'))
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


@pytest.mark.parametrize('config', 
                         [(CONFIG)])
def test_process_files(tmp_directory, download_directory, config):
    cfg = OmegaConf.create(config)

    preproc = Zinc15Preprocess(root_directory=tmp_directory)
    preproc.process_files(links_file=cfg.links_file, 
                          download_dir=download_directory,
                          pool_size=cfg.pool_size,
                          max_smiles_length=cfg.max_smiles_length)
    
    with open(cfg.links_file, 'r') as fh:
        expected_tranche_files = fh.readlines()

    tranche_files = glob.glob(os.path.join(download_directory, '*.txt'))
    assert len(tranche_files) == len(expected_tranche_files)

    expected_tranche_files = sorted([os.path.basename(x.strip()) 
                                     for x in expected_tranche_files])
    tranche_files = sorted([os.path.basename(x)
                            for x in tranche_files])
    assert tranche_files == expected_tranche_files


@pytest.mark.parametrize('config, header, hash_dict', 
                         [(CONFIG, HEADER, TRAIN_VAL_TEST_HASHES)])
def test_prepare_dataset(tmp_directory, download_directory, output_directory, config, header, hash_dict):
    cfg = OmegaConf.create(config)

    preproc = Zinc15Preprocess(root_directory=tmp_directory)
    preproc.prepare_dataset(max_smiles_length=cfg.max_smiles_length,
                            train_samples_per_file=cfg.train_samples_per_file,
                            val_samples_per_file=cfg.val_samples_per_file,
                            test_samples_per_file=cfg.test_samples_per_file,
                            links_file=cfg.links_file,
                            seed=cfg.seed)
    
    expected_lines = 0
    tranche_files = glob.glob(os.path.join(download_directory, '*.txt'))
    for file in tranche_files:
        with open(file, 'r') as fh:
            expected_lines += len(fh.readlines()) - 1
    
    total_lines = 0
    for split in ['train', 'val', 'test']:
        split_directory = os.path.join(output_directory, split)
        assert get_directory_hash(split_directory) == hash_dict[split]

        csv_file_list = sorted(glob.glob(os.path.join(split_directory, '*.csv')))
        expected_samples = cfg.get(f'{split}_samples_per_file')

        for file in csv_file_list:
            with open(file, 'r') as fh:
                lines = fh.readlines()
                num_samples = len(lines) - 1
                if file != csv_file_list[-1]:
                    assert num_samples == expected_samples
                assert lines[0].strip() == header
                total_lines += num_samples

    assert expected_lines == total_lines
