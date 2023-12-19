# Inference with NVIDIA Triton Inference Server

This section will use the pre-trained BioNeMo checkpoints to demonstrate NVIDIA [Triton Inference Server](https://github.com/triton-inference-server).
[PyTriton](https://github.com/triton-inference-server/pytriton), which is a Flask/FastAPI-like interface that simplifies Triton deployment in Python environments will be used.


## Prerequisites

* Linux OS
* Pascal, Volta, Turing, or an NVIDIA Ampere architecture-based GPU.
* NVIDIA Driver
* CUDA toolkit installed
* Docker


## Using Predefined Server-Client Scripts

BioNeMo comes with a set of example scripts for inference with PyTriton. If you would like to skip to using these, see the 
[BioNeMo PyTriton README](./bionemo/triton/README.md) file for instructions and detailed documentation.


## Detailed Example with ESM-1nv

PyTriton provides a light-weight wrapper that allows you to set up the Triton Inference Server based on existing inference code. The only requirement is that inference is done by a function, that takes as an input and returns numpy arrays of supported types (numerical types or bytes).

In this example, the **Sequence to Embedding** task for ESM1 will be used as an example. The solution will consist of two components - server that performs the inference, and a client that queries this server.

The `bionemo.model.protein.esm1nv.infer.ESM1nvInference` class provides `seq_to_embeddings` method that can be used for this purpose. This method requires a list of FASTA sequences as input (list of strings) and returns a torch Tensor object as a result, so a converter must be implemented.

On the client side, a function that takes list of sequences as input and converts it into a numpy bytes array must be implemented:

```python
import numpy as np

sequences = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVL', 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLA']

sequences = np.array([[seq] for seq in sequences])
sequences = np.char.encode(sequences, "utf-8")
```

On the server side, an inference callable that performs the following must be implemented:

* accepted input in a supported format (numpy bytes array)
* decodes it to a list of strings
* runs inference with the pre-trained BioNeMo model (for example, ESM1)
* converts output to a supported format
* and sends it back to the client

Mark this callable with the `@batch` decorator from PyTriton. This decorator converts the input request into a more suitable format that can be directly passed to the model (refer to more details on batch decorator in the [PyTrtion documentation](https://github.com/triton-inference-server/pytriton/blob/main/docs/decorators.md#batch)).

An example inference callable is provided below:
```python
from typing import Dict

import numpy as np
from pytriton.decorators import batch

model: ESM1nvInference = ...

@batch
def infer_fn(sequences: np.ndarray) -> Dict[str, np.ndarray]:
    sequences = np.char.decode(sequences.astype("bytes"), "utf-8")
    sequences = sequences.squeeze(1).tolist()

    embedding = model.seq_to_embeddings(sequences)

    response = {"embeddings": embedding.cpu().numpy()}
    return response
```

Now, define and start the server:
```python
from pytriton.model_config import Tensor
from pytriton.triton import Triton

with Triton() as triton:
    triton.bind(
        model_name="ESM1",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequences", dtype=bytes, shape=(1,)),
        ],
        outputs=[
            Tensor(name="embeddings", dtype=np.float32, shape=(-1,)),
        ],
    )

    triton.serve()
```

:::{note}
The expected shapes for the inputs and outputs are defined in `infer_fn` (without the batch dimension), where -1 denotes a dynamic size.
:::

:::{warning}
When using the `@batch` decorator, it is **vital** that the `infer_fn` parmaeter names align exactly with what is 
deinfed for `inputs` to the `.bind()` call. These names are how PyTriton ensures that the right tensors are passed
along. Similiarly, the keys in the returned dictionary must align 1:1 with the names defined in the output tensors.
:::

When the server is running, use the client to perform a query:
```python
from pytriton.client import ModelClient

with ModelClient("localhost", "ESM1") as client:
    result_dict = client.infer_batch(sequences)
```


## Extending These Examples

1. Inference callable can contain any Python code. Extend the existing example with a custom post-processing or implement a more complex, multi-step inference pipeline.

2. For more control over inference parameters (for example, sampling strategy for MegaMolBART), they can be exposed to the user. Remember to represent all inputs and outputs to the inference callable as numpy arrays.

3. Use one of the provided components (server or client) alone - they are fully compatible with native solutions for Triton Inference Server.
* Query the server with a different tool, like you would do with any other Triton instance
* Use the client to interact with any Triton server, not necessarily set up with PyTriton

1. Finally, PyTriton provides variety of options to customize the server. Refer to the [PyTriton documentation](https://triton-inference-server.github.io/pytriton/0.1.5/).
