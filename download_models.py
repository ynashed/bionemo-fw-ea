"""Script to download pretrained models from NGC or PBSS. """
import argparse
import os
from subprocess import PIPE, CalledProcessError, Popen

import yaml


ALL_KEYWORD = "all"
DATA_SOURCE_CONFIG = "artifact_paths.yaml"


def streamed_subprocess_call(cmd, verbose=False):
    """
    Run a command in a subprocess, streaming its output and handling errors.

    Args:
        cmd (str): The bash command to be executed.
        verbose (bool, optional): If True, print the command's stdout during execution.

    Raises:
        CalledProcessError: If the subprocess exits with a non-zero return code.

    Note:
        This function uses subprocess.Popen to run the specified command.
        If `verbose` is True, the stdout of the subprocess will be streamed to the console.
        ANSI escape sequences used by certain commands may interfere with the output.

    Example:
        >>> streamed_subprocess_call("ls -l", verbose=True)
        Running command: ls -l
        total 0
        -rw-r--r-- 1 user user 0 Dec 10 12:00 example.txt
        Done.
    """
    print(f"Running command: {cmd}")
    with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        print("")
        if verbose:
            for line in p.stdout:
                # TODO: ngc uses ANSI escape sequences to inline-refresh their console logs,
                # and it doesn't play well with using \r in python. Need to figure out how to
                # prevent logspam here.
                print(line)
        p.wait()

        if p.returncode != 0:
            print("STDERR out:")
            print(p.stderr.read())
            raise CalledProcessError(p.returncode, p.args)
        else:
            print("\nDone.")


def main():
    """
    Script to download pretrained checkpoints from PBSS (SwiftStack) or NGC.

    After the models are downloaded, symlinked paths are created. The models and symlinks
    are all defined in DATA_SOURCE_CONFIG.
    """

    with open(DATA_SOURCE_CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)
    all_models_list = list(config['models'].keys())
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
        choices=['ngc', 'pbss'],
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
        for model in download_list:
            model_source_path = config['models'][model][args.source]
            if args.source == "ngc":
                # NGC seems to always download to a specific directory that we can't
                # specify ourselves
                ngc_dir = os.path.split(model_source_path)[1].replace(":", "_v")
                ngc_dir = os.path.join(args.download_dir, ngc_dir)
                command = f"mkdir -p {args.download_dir} && ngc registry model download-version {model_source_path} --dest {args.download_dir} && mv {ngc_dir}/* {args.download_dir}/ && rm -d {ngc_dir}"
            elif args.source == "pbss":
                command = f"aws s3 cp {model_source_path} {args.download_dir}/ --endpoint-url https://pbss.s8k.io"
            streamed_subprocess_call(command, args.verbose)
            # Create symlinks
            source_file = config['models'][model]['symlink']['source']
            target_file = os.path.join(args.download_dir, config['models'][model]['symlink']['target'])
            target_dir = os.path.split(target_file)[0]
            command = f"mkdir -p {target_dir} && ln -sf {source_file} {target_file}"
            streamed_subprocess_call(command, verbose=True)


if __name__ == "__main__":
    main()
