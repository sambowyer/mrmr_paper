import os
import numpy as np
import joblib as jbl
from multiprocessing import Pool, cpu_count

from zarth_utils.config import Config
from method_runner import MethodRunner
from plot_matrix import ErrorMatrixPlotter

from data_utils import (
    helm_datasets, 
    openllm_datasets, 
    continuous_cat_main_datasets,
    pass_at_k_open_datasets,
)
from benchpred import all_methods

def _mrmr_method_name(k_prefix: str, order: int, objective: str) -> str:
    if int(order) == 3:
        return f"{k_prefix}mrmr_{objective}_y"
    return f"{k_prefix}mrmr{int(order)}_{objective}_y"


binary_mrmr = [
    _mrmr_method_name(k_prefix, order, "MIQ")
    for order in [3, 4, 5, 6, 7, 8, 9]
    for k_prefix in ("", "k")
]

continuous_mrmr = [
    _mrmr_method_name(k_prefix, order, "PMIQ")
    for order in [3, 4, 5, 6, 7, 8, 9]
    for k_prefix in ("", "k")
]

binary_irt = ["gpirt1", "gpirt5", "gpirt"]
continuous_irt = ["Beta_gpirt1", "Beta_gpirt5", "Beta_gpirt"]

other_baslines = [
    "metabench",
    "lasso",
    "random_search_and_learn",
    "krandom_search_and_learn",
    "random_sampling_and_learn",
    "krandom_sampling_and_learn",
    "random_sampling",
]
other_baselines = other_baslines

anchor_points = ["anchor_points_weighted", "anchor_points_predictor"]

DEFAULT_METHODS = (
    binary_mrmr
    + continuous_mrmr
    + binary_irt
    + continuous_irt
    + other_baslines
    + anchor_points
)

methods_to_run = list(DEFAULT_METHODS)

# Backward-compatible aliases used by sweep scripts.
sweep_binary_MI_y_methods = binary_mrmr
sweep_continuous_PMI_y_methods = continuous_mrmr
sweep_passk_PMI_y_methods = continuous_mrmr

subset_datasets = ['ifeval', 'musr', 'openllm_math']
# subset_datasets = ['arc_challenge', 'bbh', 'gpqa', 'mmlu_pro']

# Configuration with added parameters for flexibility
config = Config(
    default_config_dict={
        "data_source": "openllm",  # "openllm", "helm", or "glue"
        # "datasets": subset_datasets,  # List of datasets to run
        "datasets": openllm_datasets,  # List of datasets to run
        # "datasets": ["mmlu_pro"],  # List of datasets to run
        "dir_results": "./results",
        "exp_suffix": "",
        "coreset_size": "100",
        "methods": methods_to_run,  # Methods to run
        "model_split_method": "interpolation",  # "interpolation", "extrapolation", "easier_extrapolation", "binned_interpolation", "stratified", or "timing"
        # "model_split_method": "extrapolation",  # "interpolation", "extrapolation", or "easier_extrapolation"
        # "model_split_method": "easier_extrapolation",  # "interpolation", "extrapolation", or "easier_extrapolation"
        "num_train_models": "default",  # "default" or an integer for ablation
        "seed_start": 1,
        "num_run": 1, #10,  # Number of runs
        "multi_process": True,
        "use_git": False,  # Set to True if repository is clean
        "no_pass_at_k_fill": False,
        "plot_output": "individual_experiment_figures",
    },
    use_argparse=True,
)

# Override methods
if config.methods == ['default']:
    config.methods = DEFAULT_METHODS
elif config.methods == ['binary_mrmr']:
    config.methods = binary_mrmr
elif config.methods == ['continuous_mrmr']:
    config.methods = continuous_mrmr
elif config.methods == ['binary_irt']:
    config.methods = binary_irt
elif config.methods == ['continuous_irt']:
    config.methods = continuous_irt
elif config.methods == ['other_baslines'] or config.methods == ['other_baselines']:
    config.methods = other_baslines
elif config.methods == ['anchor_points']:
    config.methods = anchor_points
elif config.methods == ['sweep_binary_MI_y_methods']:
    config.methods = sweep_binary_MI_y_methods
elif config.methods == ['sweep_continuous_PMI_y_methods']:
    config.methods = sweep_continuous_PMI_y_methods
elif config.methods == ['sweep_passk_PMI_y_methods']:
    config.methods = sweep_passk_PMI_y_methods
elif config.methods == ['binary_methods']:
    config.methods = binary_mrmr + binary_irt + other_baslines + anchor_points
elif config.methods == ['continuous_methods']:
    config.methods = continuous_mrmr + continuous_irt + other_baslines + anchor_points
elif config.methods == ['all_active_methods']:
    config.methods = list(DEFAULT_METHODS)

    
# Override datasets
if config.datasets == ['openllm_datasets']:
    config.datasets = openllm_datasets
    config.data_source = 'openllm'
elif config.datasets == ['helm_datasets']:
    config.datasets = helm_datasets
    config.data_source = 'helm'
elif config.datasets == ['continuous_cat_main_datasets']:
    config.datasets = continuous_cat_main_datasets
    config.data_source = 'continuous_cat'
elif config.datasets == ['pass_at_k_open_datasets']:
    config.datasets = pass_at_k_open_datasets
    config.data_source = 'pass_at_k_open'


# Print the config
print("Config:")
print(config.to_dict())
print("Datasets:")
print(config.datasets)
print("Methods:")
print(config.methods)
print("Model split method:")
print(config.model_split_method)
print("Seed start:")
print(config.seed_start)
print("Number of runs:")
print(config.num_run)
print("Multi process:")
print(config.multi_process)
print("Coreset size:")
print(config.coreset_size)
print("Number of train models:")
print(config.num_train_models)

# Create directories
os.makedirs(config.dir_results, exist_ok=True)
os.makedirs(config.plot_output, exist_ok=True)

# Initialize runner and run experiments
runner = MethodRunner(config)
results = runner.run_all_datasets()

# MAKE_RESULTS = True
MAKE_PLOTS = False
if MAKE_PLOTS:
    # Generate plots
    plotter = ErrorMatrixPlotter(results, config.plot_output)
    plotter.plot_heatmap()
    plotter.plot_detailed_matrix()

    print("Experiment completed! Results saved to:")
    print(f"- Results: {config.dir_results}")
    print(f"- Plots: {config.plot_output}")
else:
    print("Experiment completed! Results saved to:")
    print(f"- Results: {config.dir_results}")
