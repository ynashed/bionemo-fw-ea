# esm2nv readme for internal use

The goal of this readme is to collect instructions useful for on-boarding 
Nvidia-internal developers to the openfold project.

## (1) Data preprocessing
Extract sample training data and preprocess dataset.

Note that ESM2 pretraining involves splitting the dataset on uniref50 but the training sequences are sampled from members of the clusters in uniref90. Validation and testing are still performed on uniref50 level. Hence, the model is not pretrained from a single fasta file as in ESM-1b and ESM-1v.
```bash
# unzip sample data
cd examples/tests/test_data && \
unzip uniref202104_esm2_qc_test200_val200.zip && \
cd ${BIONEMO_HOME}

# preprocess dataset
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  ++do_training=False \
  ++model.data.val_size=500 \
  ++model.data.test_size=100 \
  ++model.data.train.uf50_datapath=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/uniref50_train_filt.fasta \
  ++model.data.train.uf90_datapath=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/ur90_ur50_sampler.fasta \
  ++model.data.train.cluster_mapping_tsv=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/mapping.tsv \
  ++model.data.dataset_path=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200
```

## (2) Pretraining
Pretraining ESM2 on sample dataset. 

```bash
# training on multiple devices with tensor model parallelism
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  --config-name=pretrain_esm2_8M \
  ++do_training=True \
  ++do_testing=False \
  ++model.data.dataset_path=examples/tests/test_data/uniref202104_esm2_qc_test200_val200 \
  ++trainer.devices=2 \
  ++model.tensor_model_parallel_size=2 \
  ++model.micro_batch_size=4 \
  ++trainer.max_steps=100 \
  ++trainer.val_check_interval=10 \
   ++exp_manager.create_wandb_logger=False \
  ++exp_manager.checkpoint_callback_params.save_top_k=5
#   ++model.dwnstr_task_validation.enabled=True \
#   ++model.dwnstr_task_validation.dataset.dataset_path=examples/tests/test_data/protein/downstream
```

Known issue(s)
* OOM locally when turning on downstream task validation even on the most conservative setting

## (3) FLIP benchmark
FLIP is a collection of tasks to benchmark the expressive-ness of embeddings from a protein language model (pLM). Christian Dallago et al. published the [preprint](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v2.abstract) back in 2022. The goal is to identify a pLM such that a downstream model built on top of these sequence embedding can have maximal performance.

### (3.1) Download fasta directly from FLIP public source
Different from FLIP preprocessed dataset in FLIP validation during pretraining, we need the sequences in fasta format un-processed. We can download the raw sequences from [public source](http://data.bioembeddings.com/public/FLIP/fasta).
```bash
PWD=`pwd`
DATA_PATH="${BIONEMO_HOME}/data/FLIP"
mkdir -p ${DATA_PATH}

cd ${DATA_PATH} && \
wget http://data.bioembeddings.com/public/FLIP/fasta/all_fastas.zip && \
unzip all_fastas.zip && \
rm all_fastas.zip && \
cd ${PWD}
```

### (3.2) Extract ESM2 embeddings for specific FLIP task
Next, we extract the embeddings of the input sequences, specified by task, using `bionemo/model/infer,py`.

```bash
# Choose fasta input for the specific FLIP task
# secondary_structure/sequences.fasta
# scl/mixed_soft.fasta               
# conservation/sequences.fasta       
# bind/sequences.fasta               
# sav/mixed.fasta                    
# meltome/mixed_split.fasta          
# gb1/two_vs_rest.fasta              
# aav/seven_vs_many.fasta

FLIP_DATA_FILE=secondary_structure/sequences.fasta
TASK_NAME=$(echo $FLIP_DATA_FILE | tr "/" "\n" | head -n 1)

NEMO_PATH=esm2nv_650M_converted.nemo  # public weight from from download_models.py
DATA_PATH="${BIONEMO_HOME}/data/FLIP"
RESULT_PATH="${BIONEMO_HOME}/results"
DATA_FILE=${DATA_PATH}/${FLIP_DATA_FILE}
OUT_FILE=${RESULT_PATH}/${FLIP_DATA_FILE}

mkdir -p ${RESULT_PATH}

# inference - dump embedding in hdf5 format
# warning: not tested on multi-node environment
# warning: depend on MR: https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/merge_requests/999
python -m bionemo.model.infer \
    --config-dir ${BIONEMO_HOME}/examples/protein/esm2nv/conf \
    --config-name infer.yaml \
    trainer.num_nodes=__NUMBER_OF_NODES___ \
    trainer.devices=__NUMBER_OF_GPUS__ \
    ++model.data.output_fname="${OUT_FILE}" \
    ++model.data.dataset_path="${DATA_FILE}" \
    ++model.downstream_task.restore_from_path="${NEMO_PATH}" \
    ++model.tokenizer.model_name="facebook/esm2_t33_650M_UR50D" \
    ++trainer.precision=32 \
    ++model.data.modify_percent=0 \
    ++model.data.output_format="h5"  # new feature on MR 999
```

# bionemo-service-benchmark
Before using bionemo-service-benchmark, users need to first clone the [repo](https://gitlab-master.nvidia.com/clara-discovery/bionemo-service-benchmarking) and pull/build the docker image locally. The benchmark will automatically train and evaluate a downstream model. This is different from `cfg.model.dwnstr_task_validation.enabled` in `examples/protein/esm2nv/pretrain.py` where the pretrained weights are passed into the benchmark.

To obtain performance metrics from the embeddings:
```bash
BIONEMO_MOUNT=""
SERVICE_BENCHMARK_MOUNT=""
TASK_NAME="secondary_structure"

# OUT_EMBED_H5=/bionemo/results/${TASK_NAME}/sequences.fasta_embeddings.h5
OUT_EMBED_H5="/bionemo/results/${TASK_NAME}/sequences.fasta_hiddens.h5"

docker run -it --name bionemo-service-benchmark --gpus all --rm \
    -v ${BIONEMO_MOUNT}:/bionemo \
    -v ${SERVICE_BENCHMARK_MOUNT}:/bionemo-service-benchmarking \
    nvcr.io/nvidian/cvai_bnmo_trng/bionemo-service-benchmarking:latest \
    bash -c """
cd /bionemo-service-benchmarking/e2e_bench/protein_embedding/
/home/root/mambaforge/envs/benchmarking/bin/python ../../run.py \
    --config-name ${TASK_NAME} \
    ++analysis_hooks=[contrib.analysis_hooks.html.HTMLTableHook] \
    ++benchmark_data.schema_data.embeddings_file=${OUT_EMBED_H5} \
    ++output.path=${OUT_EMBED_H5}.json
"""
```

Users can then compare the metrics in the output json to the golden values on [this confluence page](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=CLD&title=Protein+Sequence+Predictive+Benchmarks).

Additional comments:
1. Apparently secondary_structure task is a more stable metric compared to others due to larger amount of data.

## References
For the ESM-2 language model and ESMFold:
```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
