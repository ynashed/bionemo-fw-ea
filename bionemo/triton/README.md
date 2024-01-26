# BioNeMo Model Dev Triton Server
This document shows how you can easily deploy a BioNeMo model for inference. These servers are suitable for
demonstration and development. They won't be as fast as bare-metal Triton, even with model navigator support.
However, they provide the same externally accessible  named tensor API as Triton. And they are suitable for
getting up-to-speed with Triton serving and development.

For this purpose you will use [PyTriton](https://github.com/triton-inference-server/pytriton) -- a Flask/FastAPI-like
interface that simplifies Triton's deployment in Python environments. The library allows serving Machine Learning 
models directly from Python through NVIDIA's [Triton Inference Server](https://github.com/triton-inference-server).



# Prerequisites
You need to:
- download all models (`./launch.sh download`)
- download all test data (`./launch.sh download_test_data`)

Which you can perform in one step:
```bash
./launch.sh download_all
```

If you are on a workstation and will be starting and stopping a dev container, we **strongly** recommend that you
download this data first _onto your host machine_. You should then always mount this data into your container.



# Supported Models
All base encoder-decoder bionemo models are supported. All Triton and Model Navigator-using scripts use the 
`--config-path` CLI argument to specify a local directory path containing the saved model configuration.

Examples of supported models are below. Use the `C` env var for compatibility with documented example commands:
- MegaMolBART: `CONFIG_PATH="${BIONEMO_HOME}/examples/molecule/megamolbart/conf"`
- ESM1nv: `CONFIG_PATH="${BIONEMO_HOME}/examples/protein/esm1nv/conf/"`
- ESM2nv: `CONFIG_PATH="${BIONEMO_HOME}/examples/protein/esm2nv/conf/"`
- ProtT5nv: `CONFIG_PATH="${BIONEMO_HOME}/examples/protein/prott5nv/conf"`



# Starting the Dev Model Servers
The `bionemo.triton` package provides scripts to use Triton to serve every base encoding-decoding bionemo model.
Triton provides HTTP and gRPC based APIs for each model and serving component. 

```bash
python -m bionemo.triton.{embedding,hidden,sampling}_server --config-path /path/to/dir/with/inference/conf
```

Under the hood, the scripts use `hydra` and load model configuration from `infer.yaml` present in the specified 
config directory, so you can provide custom configuration by specifying a different yaml or overriding particular 
arguments.

## Embedding Server
You can start an HTTP or gRPC server for generating embeddings as:

```bash
python -m bionemo.triton.embedding_server --config-path ${CONFIG_PATH}
```

## Hidden State Server
You can start an HTTP or gRPC server for generating the model's internal hidden states as:

```bash
python -m bionemo.triton.hidden_server --config-path ${CONFIG_PATH}
```

## Sampling Server
You can start an HTTP and gRPC server for sampling new sequences as:
```bash
python -m bionemo.triton.samplings_server --config-path ${CONFIG_PATH}
```

## Decoding Server
You can start an HTTP or gRPC server for decoding the original sequence from a model's masked hidden state representation as:

```bash
python -m bionemo.triton.decodes_server --config-path ${CONFIG_PATH}
```

NOTE: Only MegaMolBART implements sampling and decoding behavior. All models implement embedding and hidden inference capabilities.



# Interacting with the Server

## `client_encode`
You may use the `client_encode` for the `embedding`, `hidden`, and `sampling` servers:
```bash
python -m bionemo.triton.client_encode --help
```

For example:
```bash
python -m bionemo.triton.client_encode --sequences "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --sequences "c1ccccc1CC(O)=O"
```
if you loaded MegaMolBART in the server script, or:

```bash
python -m bionemo.triton.client_encode --sequences "MTADAHWIPVPTNVAYDALNPGAPGTLAFAAANGWQHHPLVTVQPLPGVVFRDAAGRSRFTQRAGD"
```
if you're serving one of the protein models, such as ESM1nv, ESM2nv, or ProtT5.


## `client_decode`
If you are using the `decode` server, you must use a different client:
```bash
python -m bionemo.triton.client_decode --help
```

For example, if you save the hidden-state output from MegaMolBART, obtained by starting `hiddens_server` and querrying with `client_encode`:
```bash
python -m bionemo.triton.client_encode --sequences "MTADAHWIPVPTNVAYDALNPGAPGTLAFAAANGWQHHPLVTVQPLPGVVFRDAAGRSRFTQRAGD" --output hiddens.json
```

Then you can decode back into the original sequence by starting a `decodes_server` with MegaMolBART and querrying with `client_decode`:
```bash
python bionemo/triton/client_decode.py --input hiddens.json
```



# Nav: Optimized Model Runtimes 
The embeddings, sampling, and hidden state servers uses runtimes optimized by [model navigator](https://github.com/triton-inference-server/model_navigator) package for inference.
You need to prepare the model before starting the embeddings server. You need to perform this step only once.

NOTE: To use this optimized artifact in `bionemo.triton.inference_wrapper`, you must provide the `--nav` argument.

Run the conversion with:
```bash
python -m bionemo.triton.nav_embeddings_export --config-path "${CONFIG_PATH}" 
```

You can run the conversion for MegaMolBART:
```bash
python -m bionemo.triton.nav_embeddings_export --config-path /workspace/bionemo/examples/molecule/megamolbart/conf
```
...ESM1nv:
```bash
python -m bionemo.triton.nav_embeddings_export --config-path /workspace/bionemo/examples/protein/esm1nv/conf
```

...ESM2nv:
```bash
python -m bionemo.triton.nav_embeddings_export --config-path /workspace/bionemo/examples/protein/esm2nv/conf
```

...and ProtT5nv:
```bash
python -m bionemo.triton.nav_embeddings_export --config-path /workspace/bionemo/examples/protein/prott5nv/conf
```



# Notes

You can also use only one of the provided components (server or client) - they are fully compatible with bare-metal Triton.
* You can interact with the server like you would do with any other Triton instance
* You can use the client to interact with any Triton server
