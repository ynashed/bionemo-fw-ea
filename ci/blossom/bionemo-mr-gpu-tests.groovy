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
    timeout(time: 60, unit: 'MINUTES') {
        node(POD_LABEL) {
            try {
                runPipeline()
            } catch(Exception e) {
                postFailure(e)
            } finally {
                postAlways()
            }
        }
    }
}

def gitLabMasterMergeRequestCheckout (
    String source_branch_name,
    String target_branch_name) {

    withCredentials([string(credentialsId: 'GITLAB_TOKEN_GKAUSHIK_STR', variable: 'GITLAB_TOKEN')]) {
        checkout([
            $class: 'GitSCM',
            branches: [[name: "origin/" + source_branch_name]],
            doGenerateSubmoduleConfigurations: false,
            extensions: [
                [$class: 'CleanCheckout'],
                [$class: 'CheckoutOption', timeout: 20],
                [$class: 'CloneOption', honorRefspec: true],
                [$class: 'GitLFSPull'],
                [$class: 'PreBuildMerge', options: [fastForwardMode: 'FF', mergeRemote: 'origin', mergeStrategy: 'DEFAULT', mergeTarget: "${params.TARGET_BRANCH}"]],
                [$class: 'UserIdentity', options: [name: "${params.GIT_NAME}", email: "${params.GIT_EMAIL}"]]
            ],
            submoduleCfg: [],
            userRemoteConfigs:[
                [
                    name:'origin',
                    url: "https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/clara-discovery/bionemo.git",
                    refspec: "+refs/"+source_branch_name+"/head:refs/remotes/origin/"+source_branch_name+" +refs/heads/"+target_branch_name+":refs/remotes/origin/"+target_branch_name
                ]
            ]
        ])
    }
}

def runPipeline(){
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

        sh "chmod -R 777 /home/jenkins/agent/workspace"

        stage('Repository checkout') {
            gitLabMasterMergeRequestCheckout(
                "${params.BIONEMO_BRANCH}",
                "${params.TARGET_BRANCH}"
            )
        }

        stage('Run tests') {
           // sh returnStdout: true, script: "make -k -C ./tests/"
            //sh returnStdout: true, script: "pytest -v /opt/nvidia/bionemo/tests/"
            sh returnStdout: true, script: "cd $WORKSPACE/bionemo; pytest"
        }
    }
}

def postFailure(err) {
    println("postFailure() was called")
    throw err
}

def postAlways(){
    println("postAlways() was called")
}
