# MolMIM FW2NIM
WHAT? Get MolMIM NIM using FW2NIM up and running.

## Common Instructions
First, build the bionemo image and then the special molmim image:
```
./launch.sh build
commit=$(git rev-parse HEAD)
cp .dockerignore save.dockerignore
rm .dockerignore
DOCKER_BUILDKIT=1 docker build -f fw2nim_examples/molmim/Dockerfile --build-arg commit=${commit} -t nvcr.io/yrthnbrndebc/testing-fw2nim:bionemo-molmim-fw2nim-demo--${commit} .
cp save.dockerignore .dockerignore
```

Now, edit the `fw2nim_examples/molmim/fw_complete_config.yaml` file: update the `image_tag` to `nvcr.io/yrthnbrndebc/testing-fw2nim:bionemo-molmim-fw2nim-demo--${commit}`.

In that yaml file, find this section:

```yaml
deploy:
  image:
    image_name: nvcr.io/yrthnbrndebc/testing-fw2nim
    image_tag: <NEW TAG VALUE HERE !!!>
```

Make sure to replace `<NEW TAG VALUE HERE !!!>` with the **actual** tag contents you made!


## Local Run
Start in one terminal:
```bash
docker run --rm -it --net=host --gpus=all nvcr.io/yrthnbrndebc/testing-fw2nim:bionemo-molmim-fw2nim-demo--${commit}
```

Wait a few seconds for it to start. Then, perform local inference:

### Controlled Generation
```bash
curl --request POST \
    localhost:8008/generate \
    --header 'Content-Type: application/json' \
    --data-raw '{"smi":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "algorithm":"CMA-ES", "num_molecules":5, "property_name":"plogP", "minimize": true, "min_similarity": 0.1, "particles": 8, "iterations": 10, "radius": 1.0}'
```

### Embedding
```bash
curl --request POST \
    localhost:8008/embedding \
    --header 'Content-Type: application/json' \
    --data-raw '{"sequences": [1,20,100, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]}'
```

### Hiddens
```bash
curl --request POST \
    localhost:8008/hidden \
    --header 'Content-Type: application/json' \
    --data-raw '{"sequences": [1,20,100, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]}' | jq '.' -c > local-hidden
```

### Decode
Must run `/hidden` first!
```bash
curl -X POST \
    localhost:8008/decode \
    --header 'Content-Type: application/json' \
    --data '@./local-hidden'
```

### Sampling
```bash
curl -X POST \
    localhost:8008/sampling \
    --header 'Content-Type: application/json' \
    --data-raw '{"sequences": ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]}'
```

## NVCF deployment
Make sure the config file is up to date!

Push the image:
```bash
docker push nvcr.io/yrthnbrndebc/testing-fw2nim:bionemo-molmim-fw2nim-demo--${commit}
```

Deploy controlled-generation (needs `fw2nim`):
```bash
pushd fw2nim_examples/molmim/ && pip install --no-deps -e . && popd
deploy-nvcf --full-config fw2im_examples/molmim/fw_complete_config.yaml --endpoint '/generate'
```

Choose `/hiddens`, `/embedding`, etc. to deploy different endpoints.

Then, for inference, get the function & version ID and `curl`:
```bash
curl --request POST \
    --url "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/${FUNCTION_ID}/versions/${VERSION_ID}" \
    --header "Authorization: Bearer ${NGC_BEARER_TOKEN}" \
    --header 'accept: application/json' \
    --header 'Content-Type: application/json' \
    --data-raw '{"smi":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "algorithm":"CMA-ES", "num_molecules":5, "property_name":"plogP", "minimize": true, "min_similarity": 0.1, "particles": 8, "iterations": 10, "radius": 1.0}'
```

Note, you need to get `NGC_BEARER_TOKEN` from the NVCF website!

Note, the URL doesn't change, expect for the function/version ID. This is why the `deploy-nvcf` tool requires you to provide _one of_ the known endpoints so it can route it correctly.
