# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

PTL_TESTS=\
	test_megamolbart_inference.py \
	test_prott5nv_inference.py \
	test_esm1nv_inference.py \
	test_dnabert_training.py \
	test_inference.py

PYTEST_FLAGS=

space=$(noop) $(noop)

test: pytest test-examples

pytest: pytest-no-lightning pytest-lightning

pytest-no-lightning:
	py.test $(PYTEST_FLAGS)$(subst $(space), --ignore , $(PTL_TESTS))

pytest-lightning:
	$(foreach test,$(PTL_TESTS),py.test $(PYTEST_FLAGS) $(test);)

test-examples:
	make -C ../examples/tests test
