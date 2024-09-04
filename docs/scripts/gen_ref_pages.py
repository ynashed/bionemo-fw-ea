# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the code reference pages and copy Jupyter notebooks and README files."""

import logging
import shutil
from pathlib import Path

import mkdocs_gen_files


# Note, constants here for now since we'll probably automate some other pulls
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent.parent
SUB_PACKAGES_DIR = ROOT / "sub-packages"
DOCS_DIR = ROOT / "docs" / "docs"
API_REFERENCE_DIR = DOCS_DIR / "API_reference"
EXAMPLES_DIR = DOCS_DIR / "examples"
DEVELOPER_GUIDE_DIR = DOCS_DIR / "developer-guide"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_api_reference() -> None:
    """Generate API reference documentation for a given source directory.

    This function iterates through all 'src' directories in the sub-packages,
    generating API reference documentation for Python files and copying Markdown files.

    Returns:
        None
    """
    sub_package_srcs = SUB_PACKAGES_DIR.rglob("src")

    for src in sub_package_srcs:
        # Process Python files
        for path in sorted(src.rglob("*.py")):
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = API_REFERENCE_DIR / doc_path
            parts = tuple(module_path.parts)

            if parts[-1] == "__init__":
                continue  # Don't generate ref pages for __init__.py
            elif parts[-1] == "__main__":
                continue  # Don't generate ref pages for __main__.py

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                identifier = ".".join(parts)
                print("::: " + identifier, file=fd)

            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT))

        # Process Markdown files
        for path in sorted(src.rglob("*.md")):
            doc_path = path.relative_to(src)
            full_doc_path = API_REFERENCE_DIR / doc_path
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.write(path.read_text())
            print(full_doc_path)
            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT))

    logging.info("API reference generation completed.")


def get_subpackage_notebooks(sub_package: Path) -> None:
    """Copy Jupyter notebooks from a sub-package to the examples directory.

    Args:
        sub_package (Path): The path to the sub-package directory.

    Returns:
        None
    """
    examples_dir = sub_package / "examples"
    if examples_dir.exists():
        for notebook in examples_dir.glob("*.ipynb"):
            logging.info(f"Found notebook: {notebook}")
            dest_dir = EXAMPLES_DIR / sub_package.name
            dest_file = dest_dir / notebook.name

            dest_dir.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(notebook, dest_file)
                mkdocs_gen_files.set_edit_path(dest_file, notebook.relative_to(ROOT))
            except IOError as e:
                logging.error(f"Failed to copy notebook {notebook}: {e}")


def get_subpackage_readmes(sub_package: Path) -> None:
    """Copy README file from a sub-package to the developer guide directory.

    Args:
        sub_package (Path): The path to the sub-package directory.

    Returns:
        None
    """
    readme_file = sub_package / "README.md"
    if readme_file.exists():
        logging.info(f"Found README: {readme_file}")
        dest_dir = DEVELOPER_GUIDE_DIR / sub_package.name
        dest_file = dest_dir / f"{sub_package.name} Overview.md"

        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(readme_file, dest_file)
            mkdocs_gen_files.set_edit_path(dest_file, readme_file.relative_to(ROOT))
        except IOError as e:
            logging.error(f"Failed to copy README {readme_file}: {e}")


def main() -> None:
    """Main function to generate documentation.

    This function orchestrates the entire process of generating API references,
    copying notebooks, and copying README files for all sub-packages.

    Returns:
        None
    """
    DOCS_DIR.mkdir(exist_ok=True)

    # generate api docs
    generate_api_reference()

    for sub_package in SUB_PACKAGES_DIR.glob("bionemo-*"):
        if sub_package.is_dir():
            logging.info(f"Processing sub-package: {sub_package}")
            get_subpackage_notebooks(sub_package)
            get_subpackage_readmes(sub_package)


if __name__ == "__main__":
    main()
