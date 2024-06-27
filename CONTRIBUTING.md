### Contributing guidelines for internal bionemo2 contributions
Note: For code review standards please see [CODE-REVIEW](CODE-REVIEW.md)


# Python Coding Standards

This page contains the Python coding standards for the BioNeMo repository. They apply to all Python code in the repository (unless external constraints prevent it).

## General Guidelines

- **Consistency, consistency, consistency.**
- **Readability**: your code should be easy to read and understand by any other engineer, including outside NVIDIA. Some tips:
  - Document your code. Make all comments complete sentences, starting with a capitalized letter and ending with a period.
  - Avoid abbreviations: 'bn' is harder to understand than 'batch_norm'.
  - Avoid baked-in constants throughout the code. Instead, specify them as parameters to your function. If you must have a constant, follow the naming guideline (e.g., `GLOBAL_CONSTANT`).
  - Avoid functions that span hundreds of lines. Large functions are more difficult to read and more difficult to test. If >120 lines, consider re-factoring it into smaller logical functions, each unit-tested and well-documented.
  - Re-use (not copy-paste) code whenever possible.
  - Usage of third-party code should be legally compatible and attributed.

## Coding Style

- We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Linting is done through `ruff` and `black`.
- Unit testing with `pytest`.
- Write Python 3 code.
- Add type annotations wherever feasible. In particular, new code should all be type-annotated as thoroughly as possible. This also obviates the need for including type hints in the function docstring.
- Max 120 chars per row, indent 4 spaces, no tabs.

To the top of each Python file please refer to the LICENSE guidelines [here](https://gitlab-master.nvidia.com/clara-discovery/infra-bionemo/-/blob/main/src/infra_bionemo/license_check.py?ref_type=heads) and see below:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Your docstring describing what this file is about."""
