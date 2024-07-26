## General Instructions
For all of these test cases, the correct docker image should be specified. They are launched as sbatch path-to-script/launch_qa_testcase_{test number}.sh.

## Test Case 1
Tests openfold training on a single node. Dataset at ${INPUT_DIR} is needed. AWS and NGC credentials are not needed.

## Test Case 2
Tests fine tuning on a single node. Dataset at ${INPUT_DIR} is needed. AWS and NGC credentials are needed.

## Test Case 3
Tests training on multiple nodes. AWS credentials are needed.

## Test Case 4
Creates a data sample on a single node. AWS credentials are needed.

## Test	Case 5
Tests inference using only sequences on a single node. AWS and NGC credentials are needed.

## Test	Case 6
Tests inference	using sequences and MSAs on a single node. AWS and NGC credentials are needed.

## Test	Case 7
Runs pre-processing on a single node. AWS and NGC credentials are needed.

## Test	Case 8
Runs intitial training with MLPerf optimization on multiple nodes. AWS credentials are needed.

## Test	Case 9
Runs initial training on a single node with the priority queue dataloader. AWS and NGC credentials are not needed.
