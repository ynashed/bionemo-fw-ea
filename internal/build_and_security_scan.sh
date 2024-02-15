#!/bin/bash
set -e
CONTAINER_SCAN_AUTH_HELPER_VERSION="v2.1.0"
CONTAINER_SCAN_HELPER_VERSION="v2.2.0"
CONTAINER_SCAN_POLICY_HELPER_VERSION="v2.2.1"
CONTAINER_SCAN_VALIDATION_HELPER_VERSION="v0.1.1"
CONTAINER_SCAN_CLI_VERSION="3.1.0" # versions < 3.0.0 are no longer supported
PSS_SSA_ID="x9thwm-cootr2q1jdv5p7b8iw4fs4ob3x6nqqsoznyk"
PSS_SSA_SCOPE="nspect.verify%20scan.anchore"
SSA_ISSUER_URL="https://${PSS_SSA_ID}.ssa.nvidia.com/token?grant_type=client_credentials&scope=${PSS_SSA_SCOPE}"
SBOM_OUTPUT_FORMAT=json
#
BIONEMO_CONTAINER_NSPECT="NSPECT-232K-DZA6"
BIONEMO_IMAGE_NAME="nvcr.io/nvidian/cvai_bnmo_trng/bionemo:latest"
TMP_IMAGE_SAVE_DIR=/tmp/security-scan-image-tar
CONTAINER_ARCHIVE_NAME=bionemo-image.tar
#
# Not sure where we got these -- Neha knows
SSA_CLIENT_ID=<FILL THIS OUT>
SSA_CLIENT_SECRET=<FILL THIS OUT>


./launch.sh build
docker pull gitlab-master.nvidia.com:5005/pstooling/pulse-group/pulse-container-scanner/pulse-cli:$CONTAINER_SCAN_CLI_VERSION

mkdir -p $TMP_IMAGE_SAVE_DIR

# Check if the file exists
if [ ! -f "$TMP_IMAGE_SAVE_DIR/$CONTAINER_ARCHIVE_NAME" ]; then
    echo "Dumping $BIONEMO_IMAGE_NAME to $TMP_IMAGE_SAVE_DIR/$CONTAINER_ARCHIVE_NAME. This could take a few minutes..."
    start_time=$(date +%s)
    docker save $BIONEMO_IMAGE_NAME > $TMP_IMAGE_SAVE_DIR/$CONTAINER_ARCHIVE_NAME
    end_time=$(date +%s)
    duration=$((end_time - start_time))


    echo "Done. Dumping image took ${duration} seconds."
else
    echo "$TMP_IMAGE_SAVE_DIR/$CONTAINER_ARCHIVE_NAME already exists, skipping image dump."
fi


echo ""
echo "Getting SSA token for image scan..."

export SSA_TOKEN=$(curl --request POST --user ${SSA_CLIENT_ID}:${SSA_CLIENT_SECRET} --header "Content-Type: application/x-www-form-urlencoded" ${SSA_ISSUER_URL} --fail | jq -r ".access_token")
if [ -z "$SSA_TOKEN" ]; then
  echo "Error fetching SSA_TOKEN. Please check that your SSA_CLIENT_SECRET is valid and not expired using the SSA Portal (https://admin.login.nvidia.com/login --> Starfleet Service Accounts --> search using SSA_CLIENT_ID)"
  exit 1
else
  echo "SSA_TOKEN set!"
fi

echo "Running security scan..."

start_time=$(date +%s)
docker run -v $TMP_IMAGE_SAVE_DIR:/security-scan-image-tar gitlab-master.nvidia.com:5005/pstooling/pulse-group/pulse-container-scanner/pulse-cli:$CONTAINER_SCAN_CLI_VERSION pulse-cli -n $BIONEMO_CONTAINER_NSPECT --ssa $SSA_TOKEN scan -i /security-scan-image-tar/$CONTAINER_ARCHIVE_NAME --sbom $SBOM_OUTPUT_FORMAT -o --output-dir=/security-scan-image-tar/results

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Done. Security scan took ${duration} seconds."

echo "Here is a list of CRITICAL and HIGH vulnerabilities:"
jq '.vulnerabilities[] | select(.severity == "High" or .severity == "Critical") | {package, fix}' $TMP_IMAGE_SAVE_DIR/results/vulns.json

