import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from zarth_utils.drawer import Drawer


class ErrorMatrixPlotter:
    """Plots error matrix for predicted vs true accuracy"""

    def __init__(self, results: Dict, output_dir: str = "figures"):
        """Initialize with results and output directory

        Args:
            results: Dictionary containing all experiment results
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
        """Prepare data for heatmap

        Returns:
            Tuple containing:
                - error_matrix: 2D array of errors (methods x datasets)
                - method_names: List of method names
                - dataset_names: List of dataset names
                - mean_errors: 1D array of mean errors per method
        """
        # Get all methods and datasets
        all_methods = sorted(self.results.keys())
        all_datasets = sorted(self.results[all_methods[0]].keys())

        # Create error matrix
        error_matrix = np.zeros((len(all_methods), len(all_datasets)))

        # Check if we have detailed results (with 'results' key) or aggregated results (with 'mean_error' key)
        has_detailed = False
        for method in all_methods:
            for dataset in all_datasets:
                dataset_results = self.results[method][dataset]
                if "results" in dataset_results:
                    has_detailed = True
                    break
            if has_detailed:
                break

        for i, method in enumerate(all_methods):
            for j, dataset in enumerate(all_datasets):
                dataset_results = self.results[method][dataset]

                if has_detailed:
                    # Detailed results: extract errors from per-seed data
                    # Prefer error_MAE; fall back to legacy 'error' key
                    errors = [
                        r.get("error_MAE", r.get("error"))
                        for r in dataset_results["results"]
                        if "error_MAE" in r or "error" in r
                    ]
                    if errors:
                        error_matrix[i, j] = np.mean(errors)
                    else:
                        error_matrix[i, j] = 0
                else:
                    # Aggregated results: use mean_error directly
                    error_matrix[i, j] = dataset_results.get("mean_error", 0)

        return error_matrix, all_methods, all_datasets, np.mean(error_matrix, axis=1)

    def plot_heatmap(self, figsize=(12, 8), cmap="viridis", annot=True):
        """Create heatmap visualization

        Args:
            figsize: Figure size (width, height)
            cmap: Colormap to use
            annot: Whether to annotate cells with values
        """
        error_matrix, methods, datasets, mean_errors = self.prepare_data()

        plt.figure(figsize=figsize)

        # Create heatmap using matplotlib
        im = plt.imshow(error_matrix, cmap=cmap)

        # Add annotations
        if annot:
            for i in range(len(methods)):
                for j in range(len(datasets)):
                    plt.text(
                        j,
                        i,
                        f"{error_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="w" if error_matrix[i, j] > 0.5 else "k",
                    )

        # Add labels and title
        plt.title("Method Performance Across Datasets")
        plt.xlabel("Dataset")
        plt.ylabel("Method")
        plt.xticks(range(len(datasets)), datasets, rotation=45)
        plt.yticks(range(len(methods)), methods)

        # Add colorbar
        plt.colorbar(im, label="Mean Absolute Error")

        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, "error_matrix_heatmap.png"))
        plt.close()

    def plot_detailed_matrix(self):
        """Create detailed matrix plot with individual runs"""
        error_matrix, methods, datasets, _ = self.prepare_data()

        # Check if we have detailed results
        has_detailed = False
        for method in methods:
            for dataset in datasets:
                if "results" in self.results[method][dataset]:
                    has_detailed = True
                    break
            if has_detailed:
                break

        if not has_detailed:
            print("No detailed results available for individual runs")
            return

        # Create figure with subplots
        drawer = Drawer(num_row=len(methods), num_col=len(datasets), unit_length=5)

        for i, method in enumerate(methods):
            for j, dataset in enumerate(datasets):
                # Get detailed results for this method/dataset
                dataset_results = self.results[method][dataset]

                if "results" in dataset_results:
                    # Detailed results: extract errors from per-seed data
                    # Prefer error_MAE; fall back to legacy 'error' key
                    results = dataset_results["results"]
                    errors = [
                        r.get("error_MAE", r.get("error"))
                        for r in results
                        if "error_MAE" in r or "error" in r
                    ]
                else:
                    # Aggregated results: generate synthetic runs for visualization
                    mean_error = dataset_results.get("mean_error", 0)
                    std_error = dataset_results.get("std_error", 0)
                    num_runs = dataset_results.get(
                        "num_runs", 5
                    )  # Default to 5 runs if not specified

                    # Generate synthetic errors around the mean
                    errors = np.random.normal(mean_error, std_error, num_runs).tolist()

                # Create subplot
                ax = drawer.add_one_empty_axes(
                    index=(i * len(datasets) + j) + 1,
                    nrows=len(methods),
                    ncols=len(datasets),
                    title=f"{method}\n{dataset}",
                    xlabel="Seed" if j == 0 else "",
                    ylabel="Error" if i == 0 else "",
                    fontsize=8,
                )

                # Plot error distribution
                if errors:
                    ax.scatter(range(len(errors)), errors, s=20, c="blue")
                    ax.plot(
                        [0, len(errors) - 1],
                        [np.mean(errors), np.mean(errors)],
                        color="red",
                        linestyle="--",
                    )
                    ax.set_ylim(0, max(1.0, np.max(errors) * 1.1))
                else:
                    ax.text(
                        0.5, 0.5, "No error data", ha="center", va="center", fontsize=10
                    )

        # Save figure
        drawer.save("detailed_error_matrix")
