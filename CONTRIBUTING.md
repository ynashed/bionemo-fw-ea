Development in BioNeMo is done through forks and merge requests (MRs). When starting to contirbute to BioNeMo, create your personal fork, add your feature branch from `dev`, commit your changes and then make a merge request into the main repo.

Don't create branches directly in the main repo.

# Merge Requests (MR) Guidelines

**Send your MRs to the `dev` branch**

1) Make sure your MR does one thing. Have a clear answer to "What does this MR do?"
2) Follow provided default MR template 
3) Make sure all unit tests finish successfully before sending MR
4) Make sure you added neccessary tests and documentation changes (could be just comments in the config files) for the feature in your MR
5) Rebase your feature branch with the latest `dev` to include any new changes that have been added. Resolve merge conflicts, if any
6) Send your MR and request a review
7) If your MR is still WIP, mark it as "Draft"
8) Your  merge request must pass all checks and be peer-reviewed before it can be merged.
9) Make sure to merge your MR when it's ready

## Unit tests
Tests must be run locally, while developing with the following command:
```
pytest
```

If your changes modify expected test results or test configs, they can be update with the following commands:

```
UPDATE_EXPECTED_RESULTS=1  pytest examples/tests/test_model_pretrain_and_downstream.py

UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py
```

After testing your code locally, test your code in the CI. Go to your merge requst -> "Pipelines" and trigger the pipeline by clicking an arrow sign.

# General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Well-tested**: please add simple, fast unit tests. 
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easy to read and well documented (with comments and docstrings).
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that BioNeMo supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.


## Python style

1. Include docstrings for every class and method exposed to the user.
1. Use Python 3 type hints for every class and method exposed to the user.
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Add ``__init__.py`` for every folder.
1. F-strings are prefered to formatted strings.
1. Loggers are preferred to print. In BioNeMo, you can use logger from ``from nemo.utils import logging``
1. Private functions (functions start with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.
 
