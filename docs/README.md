# BioNeMo Framework User Documentation

## Structure

The [contents.md](bionemo/contents.md) page creates outline for the documentation. Make sure to paths to new tutorials or high level documents to this page.

The corresponding markdown (`.md`) files are named as `xxxx-xxx-fw.md` indicating that they refer to BioNeMo FrameWork related page. This is done to distinguish them from the older BioNemo Service pages, which are retained for reference but will eventually removed from the repo.

## Developing

The `launch_docs.sh` script will create and launch the container for you:
* `launch_docs.sh build` to build the Docker container
* `launch_docs.sh dev` to run the container and execute `recompile_html.sh` inside the container. This will regularly recompiles the html as well as serving it to port 8080.

Note that you can run the above at the same time as you have your regular bionemo container attached with vscode as it sends output to another port (8080 vs 8888). You only need to modify your `ssh` config if you want to port forward both 8080 and 8888 to your [localhost for viewing](http://localhost:8080/)

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
3. Add tutorials to bionemo/notebooks
4. Tutorials -- see examples in tutorials-fw.md
5. Update the table in index.md
7. Update links in contents.md to point to any new data/notebooks/models.

#### Tips and tricks
* Note that spinx will render things differently than you see them in raw markdown format. The only way to see how things will appear to end-users is [by rendering and viewing as described in the development section](#developing).
* If you have a `.ipynb` notebook that has a cell that produces a lot of gross output that you do not want to display to the end user, you can actually hide it. To do this, you need to add a tag to the cell where you want either the input or output hidden. In VSCode you can do this by clicking the `...` on the cell, and `add tags` then type `hide-output` and hit [ENTER].
* Errors in rendering documentation will not result in errors in CI. Please look at the logs during either local or remote documenting rendering ([eg as described in the devloping section](#developing)) to make sure there are not orphaned documents, links to nowhere, misformatted bibliography entries, etc.
