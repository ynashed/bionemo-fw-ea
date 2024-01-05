# BioNeMo Framework User Documentation

## Structure

The [contents.md](bionemo/contents.md) page creates outline for the documentation.

The corresponding markdown (`.md`) files are named as `xxxx-xxx-fw.md` indicating that they refer to BioNeMo FrameWork related page. This is done to distinguish them from the older BioNemo Service pages, which are retained for reference but will eventually removed from the repo. 

## Developing

The `launch_docs.sh` script will create and launch the container for you:
* `launch_docs.sh build` to build the Docker container
* `launch_docs.sh dev` to run the container and execute `recompile_html.sh` inside the container. This will regularly recompiles the html

Then run the following inside the repo -- can either be inside the container or directly on the workstation: 

```bash
PORT=8080 && python3 -m http.server ${PORT} --directory build/html
```

## Uploading HTML Files

Command to upload the docs to s3:
    `aws s3 cp build/html/ s3://nv-developer-docs/bionemo-XXX/ --recursive`

Verify upload
    `aws s3 ls s3://nv-developer-docs/bionemo-XXX/`

## Guidelines to Contributing

### Merging your changes
1. Always make changes in a local branch. 
2. Create an MR and tag the right people for review. 
3. Follow the instructions above to build and view the docs in html to ensure everything renders as expected. 
4. If you are updating existing docs, please make sure to get it reviewed by original authors of that content. You can get this information from git history. If you are still unsure, you can either email/slack the broader team and ask for guidance on the right reviewer list. 
5. If you are writing new content, please make sure to add new files to contents.md. Otherwise they won't show up on the Table of Contents in html.

### Adding a new model
If you are adding content for a new model, below are the basic set of documents you need to provide. 
1. Model Card -- see examples in bionemo/models
2. Dataset Card -- see examples in bionemo/datasets
3. Tutorials -- see examples in tutorials-fw.md
4. Update the table in index.md  
