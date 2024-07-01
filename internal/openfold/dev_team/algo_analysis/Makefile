#
#	usage:
#		cd path-to-dev_team/algo_analysis
#		make test
#
#

REPO_RELATIVE_PATH=../../../../

test:
	ls $(REPO_RELATIVE_PATH)	
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_algo_metrics.py
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_msa_column_global_attention.py
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_outer_product_mean.py
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_tensor_shape_helpers.py
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_triangle_attention_metrics.py

test_wip:
	pytest $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/tests/tests_algo_analysis/test_multiplication.py

extra_msa_stack:
	python $(REPO_RELATIVE_PATH)/internal/openfold/dev_team/algo_analysis/run_algo_analysis_extra_msa_stack.py
