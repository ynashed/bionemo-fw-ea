#!/bin/sh

set -e

while [ $# -gt 0 ]; do
    KEY="$1"
    case $KEY in
        --ngc-api-key)
            NGC_KEY="$2"
            shift 2;;
        --installation-folder)
            INSTALLATION_DIR="$2"
            shift 2;;
        -*|--*=)
            echo "Error: Unsupported keyword ${KEY}" >&2
            exit 1 ;;
        *)
            echo "Error: Unsupported possitional argument ${KEY}" >&2
            exit 1 ;;
    esac
done

if [ -z "${INSTALLATION_DIR}" ] || [ -z "${NGC_KEY}" ]; then
    echo "--installation-folder and --ngc-api-key arguments must be set" >&2
    exit 1
fi
mkdir -p ${INSTALLATION_DIR}; cd ${INSTALLATION_DIR}
apt update; apt install -y wget unzip
wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip -o ngccli_linux.zip
chmod u+x ngc-cli/ngc
find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
printf "%s\n" ${NGC_KEY} json "ea-nvidia-drug-discovery (t6a4nuz8vrsr)" no-team no-ace | ./ngc-cli/ngc config set
export PATH="$(pwd)/ngc-cli:$PATH"
echo "$(pwd)/ngc-cli:$PATH"
