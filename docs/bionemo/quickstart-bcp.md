# Getting Started with BCP


NVIDIA Base Command Platform (BCP) includes a central user interface with a set of managed compute resources. It can be used to manage datasets, workspaces, and jobs, as well as users within an organization and team. This creates a convenient hub for monitoring job execution, viewing metrics and logs, and monitoring resource utilization. NVIDIA DGX Cloud is powered by Base Command Platform.

More information and CLI documentation can be found on the [BCP website](https://docs.nvidia.com/base-command-platform/index.html).

Before beginning, NGC CLI must be installed and configured to perform command-line tasks. Visit the [NGC user guide](https://docs.nvidia.com/base-command-platform/user-guide/index.html#installing-ngc-cli) for instructions.

## Training BioNeMo Models with BCP

The following reference script can be used to launch a job on a cluster running BCP from a local workstation using the NGC CLI. The [NGC web UI](https://docs.nvidia.com/base-command-platform/user-guide/index.html#jobs-and-gpu-instances) can also be used to configure and launch the job interactively.

### Data Processing on BCP
**Note:** The command below is incomplete. Please replace all `< >` with appropriate values. You can reach out to NGC customer support if you're not sure what these values should be.
```bash
ngc batch run \\
--name "megamolbart_sample_dataprocess" \\
--org <org> \\
--team <team> \\
--array-type "PYTORCH" \\
--ace <ace> \\
--instance <instance> \\
--commandline "cd /workspace/bionemo/examples/molecule/megamolbart && python pretrain.py --config-path=conf --config-name=pretrain_xsmall_span_aug do_training=False model.data.links_file='${oc.env:BIONEMO_HOME}/examples/molecule/megamolbart/dataset/ZINC-downloader-sample.txt' model.data.dataset_path=/data/zinc_csv" \\
--result /result \\
--image "nvidia/clara/bionemo-framework:1.1" \\
--workspace <workspace-id>:/data:RW
```

Below is a quick summary of some of the arguments in the command above. Run `ngc batch run --help` in your local terminal for more details.

Result: Private to a job, read-write artifact, automatically generated for each node in a job.

Workspace: Shareable read-write artifact, mountable to a job.

Ace: Accelerated Computing Environment.

Instance: Node type for running the job.

### Training on BCP
**Note:** The command below is incomplete. Please replace all `< >` with appropriate values. You can reach out to NGC customer support if you're not sure what these values should be.
```bash
ngc batch run \\
--name "megamolbart_sample_train" \\
--org <org> \\
--team <team> \\
--array-type "PYTORCH" \\
--ace <ace> \\
--instance <instance> \\
--commandline "python pretrain.py --config-path=conf --config-name=pretrain_xsmall_span_aug do_training=True model.data.dataset_path=/data/zinc_csv model.data.dataset.train=x000 model.data.dataset.val=x000 model.data.dataset.test=x000 exp_manager.exp_dir=/data/mmb_training" \\
--result /result \\
--image "nvidia/clara/bionemo-framework:1.1" \\
--workspace <workspace-id>:/data:RW
```

You can explore the contents of the Docker container locally to understand different scripts and models available.
