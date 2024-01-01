"""Script to download pretrained models from NGC or PBSS. """
import argparse
import os
import sys
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, model_validator


ALL_KEYWORD = "all"
DATA_SOURCE_CONFIG = Path("artifact_paths.yaml")
ModelSource = Literal['ngc', 'pbss']


#####################################################
# Define the structure of the DATA_SOURCE_CONFIG file
#####################################################
class SymlinkConfig(BaseModel):
    source: Path
    target: Path


class ModelConfig(BaseModel):
    ngc: Optional[str] = None
    pbss: Optional[str] = None
    symlink: Optional[SymlinkConfig] = None
    relative_download_dir: Optional[Path] = None
    extra_args: Optional[str] = None


class Config(BaseModel):
    models: Dict[str, ModelConfig]

    @model_validator(mode="after")
    def check_download_source_exists(cls, values):
        for model_name, model_config in values.models.items():
            if model_config.ngc is None and model_config.pbss is None:
                raise ValueError(f"Model {model_name} doesn't have a NGC or PBSS download path.")
        return values


#####################################################
# End config definition
#####################################################


def streamed_subprocess_call(cmd: str, stream_stdout: bool = False) -> Tuple[str, str, int]:
    """
    Run a command in a subprocess, streaming its output and handling errors.

    Args:
        cmd (str): The bash command to be executed.
        stream_stdout (bool, optional): If True, print the command's stdout during execution.

    Returns:
        (str, str, int): The stdout string, stderr string, and return code integer.
    Raises:
        CalledProcessError: If the subprocess exits with a non-zero return code.

    Note:
        This function uses subprocess.Popen to run the specified command.
        If `stream_stdout` is True, the stdout of the subprocess will be streamed to the console.
        ANSI escape sequences used by certain commands may interfere with the output.

    Example:
        >>> streamed_subprocess_call("ls -l", stream_stdout=True)
        Running command: ls -l
        total 0
        -rw-r--r-- 1 user user 0 Dec 10 12:00 example.txt
        Done.
    """
    stdout: List[str] = []
    stderr: str = ""
    print(f"Running command: {cmd}\n")
    with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            # TODO: ngc uses ANSI escape sequences to inline-refresh their console logs,
            # and it doesn't play well with using \r in python. Need to figure out how to
            # prevent logspam here.
            stdout.append(line)
            if stream_stdout:
                print(line, end="")
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            print(stderr, file=sys.stderr)
        else:
            print("\nDone.")
    return "".join(stdout), stderr, p.returncode


def get_available_models(config: Config, source: ModelSource) -> List[str]:
    """
    Get a list of models that are available from a given source.

    Args:
        config (Config): The artifacts configuration.
        source (str): The source of the models to download, "ngc" or "pbss".

    Returns:
        List: The list of models available in the given source.
    """
    available_models = []
    for model in list(config.models.keys()):
        if getattr(config.models[model], source):
            available_models.append(model)
    return available_models


def download_models(
    config: Config, model_list: List, source: ModelSource, download_dir_base: Path, stream_stdout: bool = False
) -> None:
    """
    Download models from a given source.

    Args:
        config (Config): The artifacts configuration.
        model_list (List): A list of model names to download that should be present in
                           the config.
        source (str): The source of the models to download, "ngc" or "pbss".
        download_dir_base (str): The target local directory for download.
        stream_stdout (bool): If true, stream the subprocess calls to stdout.
    """
    if len(model_list) == 0:
        raise ValueError("Must supply non-empty model list for download!")

    for model in model_list:
        model_source_path = getattr(config.models[model], source)
        if not model_source_path:
            print(f"Warning: {model} does not have a {source} URL; skipping download.")
            continue

        if config.models[model].relative_download_dir:
            complete_download_dir = download_dir_base / config.models[model].relative_download_dir
        else:
            complete_download_dir = download_dir_base

        if source == "ngc":
            # NGC seems to always download to a specific directory that we can't
            # specify ourselves
            ngc_dirname = Path(os.path.split(model_source_path)[1].replace(":", "_v"))
            ngc_dirname = complete_download_dir / ngc_dirname
            command = f"mkdir -p {str(complete_download_dir)} && ngc registry model download-version {model_source_path} --dest {str(complete_download_dir)} && mv {str(ngc_dirname)}/* {str(complete_download_dir)}/ && rm -d {str(ngc_dirname)}"
        elif source == "pbss":
            command = (
                f"aws s3 cp {str(model_source_path)} {str(complete_download_dir)}/ --endpoint-url https://pbss.s8k.io"
            )
        if config.models[model].extra_args:
            extra_args = config.models[model].extra_args
            command = f"{command} {extra_args}"

        _, stderr, retcode = streamed_subprocess_call(command, stream_stdout)
        if retcode != 0:
            raise ValueError(f"Failed to download {model=}! {stderr=}")
        # Create symlinks, if necessary
        if config.models[model].symlink:
            source_file = config.models[model].symlink.source
            target_file = complete_download_dir / config.models[model].symlink.target
            target_dir = target_file.parent
            command = f"mkdir -p {target_dir} && ln -sf {str(source_file)} {str(target_file)}"
            _, stderr, retcode = streamed_subprocess_call(command, stream_stdout=True)
            if retcode != 0:
                raise ValueError(f"Failed to symlink {source_file=} to {target_file=}; {stderr=}")


def load_config(config_file: Path = DATA_SOURCE_CONFIG) -> Config:
    """
    Loads the artifacts file into a dictionary.

    Return:
        (Config): The configuration dictionary that specifies where and how to download models.
    """
    with open(DATA_SOURCE_CONFIG, 'rt') as rt:
        config_data = yaml.safe_load(rt)

    return Config(**config_data)


def main():
    """
    Script to download pretrained checkpoints from PBSS (SwiftStack) or NGC.

    After the models are downloaded, symlinked paths are created. The models and symlinks
    are all defined in DATA_SOURCE_CONFIG.
    """
    config = load_config()
    all_models_list = list(config.models.keys())
    parser = argparse.ArgumentParser(description='Pull pretrained model checkpoints.')
    parser.add_argument(
        'model_name',
        nargs='+',
        choices=all_models_list + [ALL_KEYWORD],
        help='Name of the data to download (optional if downloading all data)',
    )
    parser.add_argument(
        '--download_dir',
        default='.',
        type=str,
        help='Directory into which download and symlink the model.',
    )

    parser.add_argument(
        '--source',
        choices=list(ModelSource.__args__),
        default='ngc',
        help='Pull model from NVIDIA GPU Cloud (NGC) or SwiftStack (internal). Default is NGC.',
    )

    parser.add_argument("--verbose", action="store_true", help="Print model download progress.")

    args = parser.parse_args()
    if args.model_name:
        if ALL_KEYWORD in args.model_name:
            download_list = all_models_list
        else:
            download_list = args.model_name
        download_models(config, download_list, args.source, Path(args.download_dir), args.verbose)
    else:
        print("No models were selected to download.")


if __name__ == "__main__":
    main()
