# Quickstart Guide
## 0. Get the docker image:

`devel` docker image is created in the CI for:
* every merge to `dev` (automatically)
* every MR (if you trigger the pipeline)

The image are tagged with `<base container tag>-devel`, so the tags are:
* `gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel`
* `nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-devel` (same image as above)
* `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel`

See [Building devel docker image manually](#building-devel-docker-image-manually) for instructions on how to build your own image.

## 1. Launch `devel` container for interactive development:

```bash
BIONEMO_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel"  # from dev
# BIONEMO_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel"  # from your MR
echo "BIONEMO_IMAGE=${BIONEMO_IMAGE}" >> .env  # or edit .env file manually
docker pull ${BIONEMO_IMAGE}  # make sure it's up to date
./launch.sh dev
```

## 2. Run performance linter:
```bash
bash $BIONEMO_HOME/internal/run_tpl.sh /path/to/your/script.py arg1=val1 +arg2=val2 ...
```

## 3. Run profilers:
```bash
TAP_MODE=auto python /path/to/your/script.py arg1=val1 +arg2=val2 ...
```



# Building Models and Wide Hardware Support

If you're building models or testing models against the wide variety of hardware `bionemo` supports, be sure to know the pitfals that exist with the TF32 floating point standard. While it is [actually a 19-bit number format](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407), TF32 supports the same numeric range as FP32 while decreasing precision. It can represent the same exponents that FP32 (IEEE 754) can, but it can only handle the same fractional values present in FP16 numbers. This makes TF32 _almost_ a drop-in replacement for FP32: any FP32 number can be represented in TF32, albeit with more precision loss.

TF32 supports significantly faster FLOPs than FP32 while usually providing the same amount of precision necessary for deep learning. However, there are **many** instances where using TF32 will fail while FP32 will not. If you see **convergence** issues during training, or training **diverging**, models performing worse than expected, or models producing invalid results, the issue _may_ be due to using TF32.

If you haven't already, be sure to [read this detailed collection of TF-32 related issues found in PyTorch](https://docs.google.com/document/d/1O1JTi33VgNykZLr6F-Qn4CphFmBYXdoxFBBoUorOYrs/edit).



# Performance analysis and optimization

`devel` docker container contains two tools that can help you to analyze and optimize performance of your model:
* [Torch Performance Linter](https://gitlab-master.nvidia.com/dl/gwe/torch_performance_linter) which analyzes your code and detects the most common inefficiencies (like tensor shapes not divisible by 8)
* [Torch Automated Profiler](https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler) which wraps your code with different profilers (Nsight, DLProf, ANNA, Kineto)

You should run these tools with short experiments, so limit dataset size and number of steps. Overrides to your default config might look like this:

```
++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2
```

Don't limit the number of steps too much to give all profilers enough steps to collect data.

Note, that profiling results are highly hardware-dependent, so make sure to run TAP on a target device (eg A100) as the final check (running on your workstation is a good starting point).


To analyze your workload with TPL, replace `python` in your command with `bash $BIONEMO_HOME/internal/run_tpl.sh`, eg:
```bash
overrides="++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2"

bash $BIONEMO_HOME/internal/run_tpl.sh /workspace/bionemo/examples/protein/prott5nv/pretrain.py ${overrides}
```

This will print information about all detected issues, the overall score for your code, a summary with issues grouped together, and path to full log. The output will look similar to this:
```bash
TPL: #3990 Torch Performance Linter Score: 8.32/10
--------------------------------- SUMMARY ---------------------------------
priority category                      name count                                                                                                                                                                                                                                    msg
      P0      AMP              AMP_AUTOCAST     6                                                                                                                                                                                PyTorch AMP (torch.cuda.amp.autocast) should be enabled
      P0     DATA          SYNC_DATA_LOADER     1                                                                                                                             use asynchronous torch.utils.data.DataLoader with num_workers > 0, detected a call with {'num_workers': 0}
      P1      CPU           OMP_NUM_THREADS     1                                                                                               detected env variable OMP_NUM_THREADS=None, set OMP_NUM_THREADS, the optimal value depends on the workload, default to OMP_NUM_THREADS=1
      P1     DATA NONPERSISTENT_DATA_LOADER     1                                                                                                                       use torch.utils.data.DataLoader with persistent_workers=True, detected a call with {'persistent_workers': False}
      P1      GPU              L2_PROMOTION     1                                                                                                                      increase maximum fetch granularity of GPU L2 cache to 128 bytes, detected cudaLimitMaxL2FetchGranularity=64 bytes
      P2      VER             TORCH_VERSION     1                                                                                                                                             minimum recommended PyTorch version is 2.0.0, currently running in version 2.0.0a0+1767026
       W  COMPUTE         TENSOR_CONTIGUOUS   408                                                                                                called torch.Tensor.contiguous on non-contiguous Tensor(device=cuda:0), data transposes are expensive and should be avoided if possible
       W  COMPUTE        UNSPECIFIED_DEVICE   127                                           called torch.empty with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W  COMPUTE        UNSPECIFIED_DEVICE   119                                          called torch.tensor with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W  COMPUTE        UNSPECIFIED_DEVICE     1                                           called torch.zeros with an unspecified device argument, explicitly set the device or globally set the default device with torch.set_default_tensor_type() to avoid unnecessary data movement
       W     EVAL         EVAL_GRAD_ENABLED   288 call to a nn.Module in eval mode returned a tensor with an attribute requires_grad=True, typically gradients are not needed in evaluation mode, run the evaluation under torch.no_grad context manager to disable gradient computation
       W      GPU               EMPTY_CACHE     3                                                                                                                                                               Forcing a cache clear can be expensive and should be avoided if possible
       W     SYNC          CUDA_SYNCHRONIZE     1                                                                                                                                          torch.cuda.synchronize() causes host-device synchronization and should be avoided if possible
       W     SYNC               TENSOR_ITEM     8                                                         called torch.Tensor.item on Tensor(device=cuda:0), torch.Tensor.item causes an implicit host-device synchronization if called on GPU tensors and should be avoided if possible
       W     SYNC               TENSOR_ITEM     4                                                            called torch.Tensor.item on Tensor(device=cpu), torch.Tensor.item causes an implicit host-device synchronization if called on GPU tensors and should be avoided if possible
---------------------------------------------------------------------------
See full log in torch_performance_linter_3990.log
```


To analyze your workload with TAP, add `TAP_MODE=auto` before your command, eg:

```bash
overrides="++model.data.dataset.train=x000 \
++model.data.dataset.val=x000 \
++model.data.dataset.test=x000 \
++trainer.max_steps=200 \
++trainer.val_check_interval=200 \
++exp_manager.create_wandb_logger=False \
++exp_manager.create_tensorboard_logger=False \
++model.dwnstr_task_validation.enabled=False \
++exp_manager.checkpoint_callback_params.always_save_nemo=False \
++trainer.limit_val_batches=2"


TAP_MODE=auto python /workspace/bionemo/examples/protein/prott5nv/pretrain.py ${overrides}
```

TAP will create a dir, by default `tap_results`, containing generated profiles. The easiest way to visualize the results is to use a dedicted container:
```bash
docker run -v <local path to TAP results>:/tap_results -p 8000-8003:8000-8003 -t gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler/viz:release
```
Then open localhost:8000 in your browser and select the profile you want to see.


For more details and instructions how to visualize and interpret the results, see
* https://gitlab-master.nvidia.com/dl/gwe/torch_performance_linter
* https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler
* GWE slack channel: #ghw-dl-gwe


# Building devel docker image manually

Building image manually is useful if:
* you introduced changes in `internal` dir locally
* you work with a custom BioNeMo build and you want `devel` version for it

If you have an MR with all your changes, you can use CI built image instead - it is created automatically for every pipeline and tagged with: `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<pipeline ID>-devel`


`devel` image is built on top of the BioNeMo image and is defined with `internal/Dockerfile-devel`.
Building this image requires you to provide `BIONEMO_IMAGE` argument, which specifies the base docker image:
```
--build-arg BIONEMO_IMAGE=<docker image tag>
```

This base image can be created by you locally with `./launch.sh build` or taken from our registry, eg:
* `gitlab-master.nvidia.com/clara-discovery/bionemo:pipeline-<CI pipeline ID>` to extend docker image created from your MR
* `gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest` to extend latest image created from `dev` branch.


As we install internal tools, you need to be connected to VPN and set up a ssh agent before building to authenticate:

```bash
eval "$(ssh-agent -s)"
find ~/.ssh/ -type f -exec grep -l "PRIVATE" {} \; | xargs ssh-add &> /dev/null
```

Then you can build the docker image as follows:
```bash
BASE_IMAGE="gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest
docker build --network host --ssh default --build-arg BIONEMO_IMAGE="$BASE_IMAGE" -t ${BASE_IMAGE}-devel -f internal/Dockerfile-devel .
```

For information about running our model training tests, see the [instructions on using JET.](./jet/README.md)