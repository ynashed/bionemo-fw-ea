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
                    stage('Build BioNeMo') {
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
                                --no-cache \
                                -f setup/Dockerfile .
                                """
                                
                                sh "docker push nvcr.io/nvidian/clara-lifesciences/bionemo_ci:latest"
                                sh "docker logout nvcr.io"
                            }
                        }
                    }

                    stage('Run tests') {
                        build job: 'bionemo-test', parameters: [
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
