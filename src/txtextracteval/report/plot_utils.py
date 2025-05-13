"""Utility functions for generating plots for the txtextracteval report."""

import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

logger = logging.getLogger(__name__)

def generate_error_distribution_plots(
    df: pd.DataFrame, 
    output_dir: str,
    metrics_to_plot: List[str] = None
) -> Dict[str, str]:
    """
    Generates box plots for specified error rate distributions (e.g., CER, WER) for each model.
    Saves plots to the output directory and returns a dictionary of plot filenames.

    Args:
        df: DataFrame containing the results. Must include 'model_identifier' and 'metrics' columns.
            The 'metrics' column should contain dictionaries with keys like 'cer', 'wer'.
        output_dir: Directory to save the generated plot images.
        metrics_to_plot: A list of metric keys (e.g., ["cer", "wer"]) to generate plots for.
                         Defaults to ["cer", "wer"] if None.

    Returns:
        A dictionary where keys are metric names (e.g., "cer_distribution_plot") 
        and values are the relative paths to the saved plot images.
        Returns an empty dict if no plots can be generated.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["cer", "wer"]

    plot_files = {}
    if df.empty or 'metrics' not in df.columns or 'model_identifier' not in df.columns:
        logger.warning("DataFrame is empty or missing required columns ('metrics', 'model_identifier') for plotting.")
        return plot_files

    # Extract metrics into a more usable format for plotting
    # Each row in all_metrics_data will be: {metric_name: value, 'model_identifier': model_id}
    all_metrics_data = []
    for _, row in df.iterrows():
        model_id = row['model_identifier']
        for metric_key, value in row['metrics'].items():
            if metric_key in metrics_to_plot and pd.notna(value):
                all_metrics_data.append({
                    'metric_name': metric_key.upper(), # e.g., CER, WER
                    'value': float(value),
                    'Model': model_id # Use 'Model' for better legend/axis titles in seaborn
                })
    
    if not all_metrics_data:
        logger.info("No valid metric data found to plot.")
        return plot_files

    metrics_plot_df = pd.DataFrame(all_metrics_data)

    for metric_name_upper in metrics_plot_df['metric_name'].unique():
        plt.figure(figsize=(10, 6))
        metric_data_for_plot = metrics_plot_df[metrics_plot_df['metric_name'] == metric_name_upper]
        
        if metric_data_for_plot.empty:
            logger.info(f"No data for metric {metric_name_upper} to plot.")
            continue

        sns.boxplot(x='Model', y='value', data=metric_data_for_plot,hue="Model", palette="Set2")
        
        plt.title(f"{metric_name_upper} Distribution by Model", fontsize=15)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric_name_upper, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_filename = f"{metric_name_upper.lower()}_distribution.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        try:
            plt.savefig(plot_path)
            logger.info(f"Saved {metric_name_upper} distribution plot to {plot_path}")
            # Store relative path for Markdown report
            plot_files[f"{metric_name_upper.lower()}_distribution_plot"] = os.path.relpath(plot_path, output_dir)
        except Exception as e:
            logger.error(f"Failed to save plot {plot_path}: {e}")
        finally:
            plt.close() # Close the figure to free memory

    return plot_files 