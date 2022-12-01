podTemplate (cloud:'sc-ipp-blossom-prod', yaml : '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: bionemo-mr-container
    image: jenkins/inbound-agent:alpine
    resources:
        requests:
            cpu: 2
            memory: 200Mi
    command: [ "/bin/bash", "-c", "--" ]
    args: [ "while true; do sleep 30; done;" ]
  nodeSelector:
    nodeType: cpu
    kubernetes.io/os: linux
''')
{
    timeout(time: 120, unit: 'MINUTES') {
        node(POD_LABEL) {
            try {
                runPipeline()
            } catch(Exception e) {
                postFailure(e)
            }
        }
    }
}

def runPipeline(){
    stage("Run tests") {
        sh "env"
        env['prTitle'] = ""
        if (env['gitlabMergeRequestIid'] != null) {
            updateGitlabCommitStatus name: 'bionemo-mr', state: 'running'
            sh 'echo gitlab MR build - gathering information from environment variables'
            env['prId'] = env['gitlabMergeRequestIid']
            env['sourceBranch'] = env['gitlabSourceBranch']
            env['targetBranch'] = env['gitlabTargetBranch']
            env['commitHash'] = 'merge-requests/'+env['gitlabMergeRequestIid']
            env['prAuthor'] = env['gitlabUserName']
            env['prTitle'] = env['gitlabMergeRequestTitle']
        } else {
            // to run job manually (not triggered by GitLab webghook)
            env['commitHash'] = 'dev'
            env['targetBranch'] = 'dev'
        }

        container('bionemo-mr-container') {
            stage("GPU MR Build") {
                build job: 'bionemo-mr-gpu-tests', parameters: [
                    string(name: 'BIONEMO_BRANCH', value: env.commitHash),
                    string(name: 'TARGET_BRANCH', value: env.targetBranch),
                    string(name: 'GIT_NAME', value: env.gitlabUserName),
                    string(name: 'GIT_EMAIL', value: env.gitlabUserEmail),
                ]
            }
                
            if (env['gitlabMergeRequestIid'] != null) {
                updateGitlabCommitStatus(name: 'bionemo-mr', state: 'success')
            }
        }
    }
}

def postFailure(err) {
    println("postFailure() was called")
    updateGitlabCommitStatus name: 'bionemo-mr', state: 'failed'
    throw err
}
