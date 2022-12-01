podTemplate (cloud:'sc-ipp-blossom-prod', yaml: '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: bionemo
    image: nvcr.io/nvidian/clara-lifesciences/bionemo_ci:latest
    imagePullPolicy: Always
    resources:
        requests:
            cpu: 2
            nvidia.com/gpu: 1
        limits:
            nvidia.com/gpu: 1
    command:
    - sleep
    args:
    - 99d
    env:
      - name: DOCKER_HOST
        value: tcp://localhost:2375
  - name: docker-daemon
    image: docker:19.03.1-dind
    imagePullPolicy: Always
    securityContext:
      privileged: true
    env:
      - name: DOCKER_TLS_CERTDIR
        value: ""
  imagePullSecrets:
    - name: gitlab-master-omosafi-username-password-docker-reg-2
  nodeSelector:
    nvidia.com/gpu_type: A100_PCIE_40GB
    kubernetes.io/os: linux
''')
{
    timeout(time: 120, unit: 'MINUTES') {
        node(POD_LABEL) {
            try {
                container('bionemo') {
                    stage('Test GPUs') {
                        sh "nvidia-smi"
                    }
                    
                    stage('Download models from NGC') {
                        withCredentials([string(credentialsId: 'NGC_TOKEN', variable: 'NGC_TOKEN')]) {
                            sh "chmod +x /opt/nvidia/bionemo/ci/blossom/utilities/setup-ngc-cli.sh"
                            sh returnStdout: true, script: "/opt/nvidia/bionemo/ci/blossom/utilities/setup-ngc-cli.sh --ngc-api-key ${NGC_TOKEN} --installation-folder \$(pwd)"
                            sh "chmod +x /opt/nvidia/bionemo/ci/blossom/utilities/download_all_models.sh"
                            sh returnStdout: true, script: "/opt/nvidia/bionemo/ci/blossom/utilities/download_all_models.sh \$(pwd)"
                        }
                    }

                    stage('Run tests') {
                        sh returnStdout: true, script: "make -k -C /opt/nvidia/bionemo/tests/"
                    }
                }
            } catch (Exception ex) {
                print(ex)
                throw ex
            }
        }
    }
}
