# Finetune pre-trained ESM2 in BioNeMo with LoRA

This section introduces an extension of BioNeMo fine-tuning of ESM2 models with Low-Rank Adaptation (LoRA).

LoRA is a parameter-efficient approach to adapt large pre-trained language to downstream tasks. LoRA addresses some of the drawbacks of full fine-tuning by freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the Transformer architecture. Instead of fine-tuning all the weights that constitute the weight matrix of the pre-trained large language model, two smaller matrices that approximate this larger matrix are fine-tuned. These matrices constitute the LoRA adapter. This approach drastically reduces the number of trainable parameters, making finetuning more efficient and less prone to overfitting, especially when the target task has limited data.

## Setup

**Before diving in, ensure you have all [necessary prerequisites](./pre-reqs.md). If this is your first time using BioNeMo, we recommend following the [quickstart guide](./quickstart-fw.md) first.**

All commands should be executed inside the BioNeMo Framework container.

## Finetune ESM2 model in BioNeMo

### Finetuning Configuration
BioNeMo framework supports easy fine-tuning on downstream tasks by loading the pretrained model that can be frozen or unfrozen and adding a task-specific head. BioNeMo also provides example config files for downstream task fine-tuning of ESM2 on some FLIP tasks.

Pre-trained ESM2 model can be provided using a path to a NeMo model (via ``restore_encoder_path``). This is done through:

* adding ```model.restore_encoder_path:``` to the config yaml
* passing ```model.restore_encoder_path:``` as a command line argument into your script

```python
python examples/protein/downstream/downstream_flip.py \
    --config-path=<path to dir of configs> \
    --config-name=<name of config without .yaml> \
    do_training=True \
    model.encoder_frozen=False \
    model.restore_encoder_path="<path to .nemo model file>"
```

For example, we can finetune ESM2 8M model on secondary struture task using BioNeMo:

```python
python examples/protein/downstream/downstream_flip.py \
    --config-path=examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    do_training=True \
    model.encoder_frozen=False \
    model.restore_encoder_path="<path to ESM2_8M .nemo model file>"
```

However, in this approach where the encoder is unfrozen, we train all the layers of the encoder and the task head. As discussed above, LoRA provides an alternative to this.

### Using LoRA

To use LoRA for ESM2, you simply add ``model.peft`` parameters to the config yaml and ensure that ``model.encoder_frozen`` is set to false.

```shell
model:
  peft:
    enabled: True # indicates whether we intend to use PEFT technique
    peft_scheme: "lora"  # currently supported: lora
    restore_from_path: null #set to null to initialize random weights and train

    lora_tuning:
      adapter_dim: 32
      adapter_dropout: 0.0
      column_init_method: 'xavier' # options: xavier, zero or normal
      row_init_method: 'zero' # options: xavier, zero or normal
      layer_selection: null  # selects in which layers to add lora adapters. e.g. [1,12] will add lora to layer 1 (lowest) and 12. null will apply adapters to all layers
      weight_tying: False
      position_embedding_strategy: null # used only when weight_tying is True
```

Now we can finetune ESM2 8M model on secondary struture task using LoRA and the included LoRA yaml in BioNeMo:

```python
python examples/protein/downstream/downstream_flip.py \
    --config-path=examples/protein/esm2nv/conf \
    --config-name=downstream_sec_str_LORA \
    do_training=True \
    model.peft.enabled=True \
    model.peft.lora_tuning.adapter_dim=16 \
    model.encoder_frozen=False \
    model.restore_encoder_path="<path to ESM2_8M .nemo model file>"
```

The parameter ``model.peft.lora_tuning.adapter_dim`` allows us to set different values for the rank used in matrix decomposition. This is a helpful hyperparameter to maximize performance on your data as it determines trainable parameters.
