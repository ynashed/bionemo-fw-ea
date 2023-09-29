# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging

from bionemo.data import UniRef50Preprocess, FLIPPreprocess
from bionemo.data.mapped_dataset import Uniref90ClusterMappingDataset
from bionemo.model.protein.esm1nv import ESM1nvModel, esm1nv_model
from bionemo.model.utils import setup_trainer
from bionemo.utils import BioNeMoSaveRestoreConnector

from bionemo.utils.callbacks.callback_utils import setup_callbacks


@hydra_runner(config_path="../../../examples/protein/esm2nv/conf", config_name="pretrain_small_esm2")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    callbacks = setup_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)
    print("CONFG BELOW====================================================")

    with open("uf90-resampling.yaml", 'w')  as fd:
        fd.write(OmegaConf.to_yaml(cfg))
    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = esm1nv_model.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer,
                save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
            # Great, now how do I verify ths is all working? do the datasets work right now?

        # NOTE this still wont quite do it
        #       this is going to let us inspect the dataset after it has been initialized, but it will not tell us what the underlying
        #       dataset should have been (e.g. the uf50 dataset AND the uf90 dataset should be yielding results for this test)
        from bionemo.model.utils import initialize_distributed_parallel_state
        initialize_distributed_parallel_state(local_rank=0, tensor_model_parallel_size=1,pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=0)
        train_ds, val_ds2, test_ds2 = esm1nv_model.ESM2nvModel._build_train_valid_test_datasets(trainer, cfg.model)
        train_ds: Uniref90ClusterMappingDataset = train_ds
        uf50 = train_ds.uniref50_dataset
        for i, (uf50, uf90) in enumerate(zip(uf50, train_ds)):
            if i > 10: sys.exit()
            print(uf50, uf90['sequence_id'], uf90['sequence'])
            assert uf50.split("UniRef50_")[1] in uf90['sequence_id'].split("UniRef90_")[1]
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        raise NotImplementedError("Preprocessing for Uniref90 not connected.")
        preprocessor = UniRef50Preprocess()
        # NOTE
        # What preprocessing steps need to happen here?
        # pulling the datasets, splitting uf50 into CSVs, and loading/creating the cluster map.
        preprocessor.prepare_dataset(ngc_registry_target=cfg.model.data.ngc_registry_target,
                                     ngc_registry_version=cfg.model.data.ngc_registry_version,
                                     output_dir=cfg.model.data.dataset_path)
        # Downloading and preprocessing data for downstream task validation
        if cfg.model.dwnstr_task_validation.enabled:
            flip_preprocessor = FLIPPreprocess()
            if "task_name" not in cfg.model.dwnstr_task_validation.dataset:
                task_name = cfg.model.dwnstr_task_validation.dataset.dataset_path.split("/")[-1]
            else:
                task_name = cfg.model.dwnstr_task_validation.dataset.task_name
            flip_preprocessor.prepare_dataset(output_dir=cfg.model.dwnstr_task_validation.dataset.dataset_path,
                                              task_name=task_name)


if __name__ == '__main__':
    main()
