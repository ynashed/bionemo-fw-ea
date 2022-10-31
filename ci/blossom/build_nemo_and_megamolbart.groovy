podTemplate (cloud:'sc-ipp-blossom-prod', yaml: '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker-dind
    resources:
        requests:
            cpu: 2
    image: docker:19.03.1
    imagePullPolicy: Always
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
  nodeSelector:
    kubernetes.io/os: linux
''')
{
    timeout(time: 120, unit: 'MINUTES') {
        node(POD_LABEL) {
            gitlabCommitStatus {
                try {                
                    stage('Build NeMo Image') {
                        container('docker-dind') {
                            sh "docker --version"      
                            checkout([
                                $class: 'GitSCM',
                                branches: [[name: "*/${params.NEMO_BRANCH}"]], 
                                doGenerateSubmoduleConfigurations: false, 
                                extensions: [
                                    [$class: 'CleanCheckout'], [$class: 'CheckoutOption', timeout: 120], [$class: 'GitLFSPull']
                                ], 
                                submoduleCfg: [], 
                                userRemoteConfigs: [[url: "https://github.com/NVIDIA/NeMo.git"]]
                            ])
                            
                            withCredentials([string(credentialsId: 'NGC_TOKEN', variable: 'NGC_TOKEN')]) {
                                sh "docker login nvcr.io -u '\$oauthtoken' -p ${NGC_TOKEN}"
                            }
                            
                            sh "DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nvcr.io/nvidian/clara-lifesciences/bionemo_training:latest ."
                            sh "docker push nvcr.io/nvidian/clara-lifesciences/bionemo_training:latest"
                        }
                    }
                    
                    stage('Build MegaMolBART image') {
                        container('docker-dind') {
                            withCredentials([string(credentialsId: 'GITLAB_TOKEN_GKAUSHIK_STR', variable: 'GITLAB_TOKEN')]) {
                                checkout([
                                    $class: 'GitSCM',
                                    branches: [[name: "*/${params.BRANCH_NAME}"]], 
                                    doGenerateSubmoduleConfigurations: false, 
                                    extensions: [
                                        [$class: 'CleanCheckout'], [$class: 'CheckoutOption', timeout: 120], [$class: 'GitLFSPull']
                                    ], 
                                    submoduleCfg: [], 
                                    userRemoteConfigs: [[url: "https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/clara-discovery/bionemo.git"]]
                                ])
                            }    

                            withCredentials([string(credentialsId: 'NGC_TOKEN', variable: 'NGC_TOKEN'), string(credentialsId: 'GITLAB_TOKEN_GKAUSHIK_STR', variable: 'GITLAB_TOKEN')]) {
                                sh "docker login nvcr.io -u '\$oauthtoken' -p ${NGC_TOKEN}"
                                
                                sh """
                                docker build --network host \
                                -t nvcr.io/nvidian/clara-lifesciences/bionemo_ci:latest \
                                --build-arg GITHUB_ACCESS_TOKEN=${GITLAB_TOKEN} \
                                --build-arg GITHUB_BRANCH=${params.BRANCH_NAME} \
                                --build-arg BASE_IMAGE=nvcr.io/nvidian/clara-lifesciences/bionemo_training:latest \
                                --no-cache \
                                -f setup/Dockerfile .
                                """
                                
                                sh "docker push nvcr.io/nvidian/clara-lifesciences/bionemo_ci:latest"
                                sh "docker logout nvcr.io"
                            }
                        }
                    }

                    stage('Call run_megamolbart_tests script') {
                        build job: 'nemo-bionemo-run-gpu-tests', parameters: [
                            string(name: 'BRANCH_NAME', value: params.BRANCH_NAME)
                        ]
                    }
                } catch (Exception ex) {
                    print(ex)
                    throw ex
                }
            }
        }
    }
}
