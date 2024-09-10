import nemo_run as run
from factories import pretrain

if __name__ == "__main__":
    # TODO see if we can setup the experiment management thingy too.
    run.cli.main(pretrain)