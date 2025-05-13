#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Markdown report generation for txtextracteval results."""

import logging
import os
from typing import List, Dict, Any
import datetime
import pandas as pd # Using pandas for easier aggregation and table formatting
import shutil # For copying config file
import difflib # For text diffs
# import matplotlib.pyplot as plt # Placeholder for plotting
# import seaborn as sns # Placeholder for plotting
from .plot_utils import generate_error_distribution_plots # Import the new plotting function

logger = logging.getLogger(__name__)

# Helper to generate slug for anchor links
def _to_slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")

def format_metric(value: Any) -> str:
    """Formats metric values nicely for the report."""
    if pd.isna(value): return 'N/A' # Handle NaN before formatting
    if isinstance(value, float):
        if 0 < abs(value) < 0.0001 or abs(value) > 1e6:
             return f"{value:.2e}" # Scientific notation for very small/large
        else:
             return f"{value:.4f}" # Standard decimal format
    return str(value)

def sanitize_for_markdown_table(text: str | None) -> str:
    """Replaces pipe characters with escaped versions to avoid breaking tables."""
    if text is None: return ""
    # Replace pipe with escaped pipe for Markdown rendering (GFM standard is \|)
    return str(text).replace("|", "\\|")

def _sanitize_for_inline_markdown_code(text: str | None) -> str:
    """Sanitizes text to be safely included within a Markdown inline code span (`text`)."""
    if text is None:
        return ""
    processed_text = str(text)
    # Replace newlines with spaces to keep the inline code on one line in the table
    processed_text = processed_text.replace('\n', ' ')
    # Escape backticks to prevent them from prematurely closing the inline code span
    processed_text = processed_text.replace('`', '\\`')
    # Escape pipe characters as this will still be within a table cell
    processed_text = processed_text.replace('|', '\\|')
    return processed_text

def _get_model_identifier(row: pd.Series) -> str:
    """Extracts a model identifier from the result row."""
    method_type = row['method_type']
    config = row['method_config']
    if method_type == 'tesseract':
        # Tesseract might have lang/psm, but the core model is just tesseract
        # lang = config.get('lang')
        # psm = config.get('psm')
        # return f"tesseract (lang={lang or 'def'}, psm={psm or 'def'})"
        return "tesseract"
    elif method_type == 'hf_ocr':
        return config.get('model', 'hf_ocr_default')
    elif method_type == 'llm_api':
        provider = config.get('provider', 'unknown_llm')
        model = config.get('model', f'{provider}_default')
        return model # Return the specific model name used by the LLM provider
    else:
        return method_type # Fallback

def _generate_text_diff(text1: str, text2: str) -> str:
    """Generates a textual diff for Markdown block."""
    if not isinstance(text1, str): text1 = str(text1)
    if not isinstance(text2, str): text2 = str(text2)
    
    diff_lines = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    # Content within ```diff ... ``` block is generally treated as literal.
    # No need to escape backticks inside the diff lines themselves.
    # Ensure the lines are simply joined by \n.
    # Add a trailing newline to ensure the closing ``` is on its own line before any subsequent HTML.
    return "```diff\n" + "\n".join(diff_lines) + "\n```\n"

# Placeholder for plotting function - to be developed
# def generate_error_distribution_plots(df: pd.DataFrame, output_dir: str) -> Dict[str, Dict[str, str]]:
#     plot_files = {}
#     # Implement plotting logic using matplotlib/seaborn
#     # Save plots to output_dir
#     # Return dict like: {"model_identifier": {"cer_plot": "path/to/cer.png", "wer_plot": "path/to/wer.png"}}
#     return plot_files

def generate_markdown_report(results: List[Dict[str, Any]], config: Dict[str, Any], output_dir: str, filename: str, config_file_path: str | None = None) -> None:
    """Generates a Markdown report from the experiment results.

    Args:
        results: List of result dictionaries from run_experiment.
        config: The original experiment configuration dictionary (for context).
        output_dir: Directory where the report and associated images are saved.
        filename: The name for the output Markdown file.
        config_file_path: Path to the original config file for copying.
    """
    report_path = os.path.join(output_dir, filename)
    logger.info(f"Generating Markdown report at: {report_path}")

    if not results:
        logger.warning("No results to generate report from.")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Text Extraction Evaluation Report\n\n")
            f.write("**Warning:** No results were generated during the experiment run.\n")
        return

    # Use pandas DataFrame
    df = pd.DataFrame(results)

    # --- Preprocess DataFrame --- # 
    # Ensure metrics is a dict
    df['metrics'] = df['metrics'].apply(lambda x: x if isinstance(x, dict) else {})
    # Ensure misc is a dict, and extract token counts if available
    if 'misc' in df.columns:
        df['misc'] = df['misc'].apply(lambda x: x if isinstance(x, dict) else {})
        df['input_tokens'] = df['misc'].apply(lambda x: x.get('input_tokens'))
        df['output_tokens'] = df['misc'].apply(lambda x: x.get('output_tokens'))
    else:
        # If 'misc' column doesn't exist at all, create empty token columns to prevent later errors
        df['input_tokens'] = pd.Series([None] * len(df), index=df.index)
        df['output_tokens'] = pd.Series([None] * len(df), index=df.index)

    # Add provider column (same as method_type for clarity in tables)
    df['provider'] = df['method_type']
    # Add model identifier column
    df['model_identifier'] = df.apply(_get_model_identifier, axis=1)

    # --- Report Content --- #
    report_content = []

    # --- Table of Contents --- #
    toc_entries = []
    overview_slug = _to_slug("Experiment Overview")
    summary_slug = _to_slug("Summary Metrics (Averages)")
    detailed_slug = _to_slug("Detailed Results")
    
    toc_entries.append(f"- [Experiment Overview](#{overview_slug})")
    toc_entries.append(f"- [Summary Metrics (Averages)](#{summary_slug})")
    toc_entries.append(f"- [Detailed Results](#{detailed_slug})")

    # Add image-specific links to TOC
    image_slugs = []
    if not df.empty:
        for i, original_image in enumerate(df['original_image_path'].unique()):
            img_basename = os.path.basename(original_image)
            img_slug = _to_slug(f"Source Image {img_basename}")
            image_slugs.append(img_slug) # Store for later use when generating headers
            toc_entries.append(f"  - [Source Image: {img_basename}](#{img_slug})")
    
    report_content.append("## Table of Contents")
    report_content.append("\n".join(toc_entries))
    report_content.append("\n\n---") # Ensure blank line before HR

    # Header
    report_content.append(f"# Text Extraction Evaluation Report")
    report_content.append(f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") # Single newline after this is okay if followed by heading

    # Experiment Overview
    report_content.append(f"## Experiment Overview")
    # Basic info from config - list items handle their own spacing mostly
    report_content.append(f"*   **Images Processed:** {df['original_image_path'].nunique()}")
    report_content.append(f"*   **Methods Tested:** {str(list(df['method_type'].unique()))}")
    report_content.append(f"*   **Transformations Applied:** {str(config.get('transformations', 'None'))}")
    report_content.append(f"*   **Metrics Calculated:** {str(config.get('metrics', []))}") # No trailing \n needed if followed by another item or blank line

    # Configuration file copy
    report_content.append("\n") # Blank line before this paragraph
    if config_file_path and os.path.exists(config_file_path):
        copied_config_filename = f"config_used_for_report_{os.path.basename(config_file_path)}"
        copied_config_path = os.path.join(output_dir, copied_config_filename)
        try:
            shutil.copy2(config_file_path, copied_config_path)
            report_content.append(f"The configuration file used for this run (`{os.path.basename(config_file_path)}`) has been copied to the output directory as `{copied_config_filename}`.")
        except Exception as e:
            logger.warning(f"Could not copy config file {config_file_path} to {copied_config_path}: {e}")
            report_content.append(f"_Note: Could not automatically copy the configuration file ({os.path.basename(config_file_path)}) to the output directory._")
    else:
        report_content.append("_Note: Original configuration file path not provided or file not found; not copied to output directory._")
    report_content.append("\n") # Ensure a blank line after this paragraph block

    # --- Summary Table --- #
    report_content.append(f"## Summary Metrics (Averages)")
    try:
        # Group by provider and model
        grouping_cols = ['provider', 'model_identifier']
        summary = df.groupby(grouping_cols).agg(
            avg_latency=('latency_seconds', 'mean'),
            avg_cost=('cost', 'mean'),
        ).reset_index()

        # === COST MODIFICATION: Multiply avg_cost by 1000 ===
        if 'avg_cost' in summary.columns:
            summary['avg_cost'] = summary['avg_cost'] * 1000
        # === END COST MODIFICATION ===

        # Unpack metrics for averaging
        metrics_df = pd.json_normalize(df['metrics']) # Flatten the metrics dict
        # Add grouping columns back to metrics_df
        for col in grouping_cols:
            metrics_df[col] = df[col]
        metrics_summary = metrics_df.groupby(grouping_cols).mean().reset_index()

        # Merge summaries
        summary = pd.merge(summary, metrics_summary, on=grouping_cols, how='left')

        # Format columns
        summary_formatted = summary.copy()
        for col in summary_formatted.columns:
            if col not in grouping_cols: # Don't format the grouping columns
                summary_formatted[col] = summary_formatted[col].apply(format_metric)

        # --- Prepare column rename mapping --- #
        # Start with static renames
        rename_map = {
            'provider': 'Provider',
            'model_identifier': 'Model',
            'avg_latency': 'Avg Latency (s)',
            'avg_cost': 'Avg Cost / 1k images ($)' # COST MODIFICATION: Update label
        }
        # Dynamically add metric column renames (e.g., 'cer' to 'Avg CER')
        metric_cols = list(metrics_summary.columns)
        for col in grouping_cols:
            if col in metric_cols: metric_cols.remove(col)
        
        # Define which metrics are 'lower is better' for highlighting
        # All default metrics (latency, cost, cer, wer) are lower is better.
        # This assumes any other custom metrics added to metrics_summary are also lower-is-better.
        # If a higher-is-better metric is introduced, this logic will need adjustment.
        lower_is_better_metric_originals = ['avg_latency', 'avg_cost'] + metric_cols

        for m_col in metric_cols:
            rename_map[m_col] = f"Avg {m_col.upper()}"
        
        summary_formatted_renamed = summary_formatted.rename(columns=rename_map)

        # Bold best performer
        styled_summary_df = summary_formatted_renamed.copy()

        for original_metric_name in lower_is_better_metric_originals:
            # Get the renamed column name for the current original_metric_name
            metric_col_renamed = rename_map.get(original_metric_name)
            if not metric_col_renamed or metric_col_renamed not in styled_summary_df.columns:
                continue # Skip if this metric isn't in the final table or wasn't renamed (should not happen for these)

            # Need to convert back to numeric for min() or max()
            # The 'summary' DataFrame holds the raw numeric averages before string formatting
            numeric_series = pd.to_numeric(summary[original_metric_name], errors='coerce')
            if not numeric_series.empty and numeric_series.notna().any():
                best_val_numeric = numeric_series.min() # Assuming lower is better
                
                # Find indices of best_val and bold them in styled_summary_df
                for idx, val_numeric in numeric_series.items():
                    if pd.notna(val_numeric) and val_numeric == best_val_numeric:
                        # summary_formatted contains the already string-formatted (but not renamed) values
                        # We use original_metric_name to get the correctly formatted value from summary_formatted
                        # Then, we apply bolding to this string in styled_summary_df using the renamed column name.
                        current_formatted_val_from_summary_formatted = summary_formatted.loc[idx, original_metric_name]
                        styled_summary_df.loc[idx, metric_col_renamed] = f"**{current_formatted_val_from_summary_formatted}**"

        report_content.append(styled_summary_df.to_markdown(index=False, disable_numparse=True))
        report_content.append("\n\n_Note: Averages include results from all image variants._\n") # Blank line before note, single after is fine

        # --- Average Metrics per Transformation ---
        report_content.append("<details>")
        report_content.append("  <summary>Average Metrics per Transformation (Click to expand)</summary>")

        # Content inside details should be appropriately spaced Markdown
        report_content.append("\n\n  _This section shows average metrics for each method, broken down by individual transformation applied to the source images._\n")

        # Create 'transformation_name' column from 'variant_desc'
        # This is a simplified assumption. Real parsing might be needed if variant_desc is complex.
        # If 'ocr_prep_denoise_ksize_3_crop_frac_0.03' then 'ocr_prep'
        # If 'blur_kernel_size_5' then 'blur'
        # If 'original' then 'original'
        df['transformation_name'] = df['variant_desc'].apply(lambda x: x.split('_')[0] if x != 'original' else 'original')

        transform_summary_dfs = {}
        for transform_name, group_df in df.groupby('transformation_name'):
            if group_df.empty: continue
            
            grouping_cols_transform = ['provider', 'model_identifier']
            current_transform_summary = group_df.groupby(grouping_cols_transform).agg(
                avg_latency=('latency_seconds', 'mean'),
                avg_cost=('cost', 'mean'),
            ).reset_index()

            if 'avg_cost' in current_transform_summary.columns:
                 current_transform_summary['avg_cost'] = current_transform_summary['avg_cost'] * 1000
            
            metrics_df_transform = pd.json_normalize(group_df['metrics'])
            for col in grouping_cols_transform:
                metrics_df_transform[col] = group_df[col].values # Align indices
            
            metrics_summary_transform = metrics_df_transform.groupby(grouping_cols_transform).mean().reset_index()
            current_transform_summary = pd.merge(current_transform_summary, metrics_summary_transform, on=grouping_cols_transform, how='left')

            current_transform_summary_formatted = current_transform_summary.copy()
            # Use the same rename_map as the main summary table for consistency
            current_transform_summary_formatted = current_transform_summary_formatted.rename(columns=rename_map)
            
            for col in current_transform_summary_formatted.columns:
                if col not in ['Provider', 'Model']: # Don't format Provider/Model
                     # Need original column name to apply format_metric correctly
                    original_col_name_found = False
                    for k, v in rename_map.items():
                        if v == col:
                            current_transform_summary_formatted[col] = current_transform_summary[k].apply(format_metric)
                            original_col_name_found = True
                            break
                    if not original_col_name_found and col in current_transform_summary.columns: # Fallback if not in rename_map (e.g. new metrics)
                         current_transform_summary_formatted[col] = current_transform_summary[col].apply(format_metric)


            transform_summary_dfs[transform_name] = current_transform_summary_formatted
        
        if transform_summary_dfs:
            for transform_name, T_summary_df in transform_summary_dfs.items():
                report_content.append(f"\n\n  **Transformation: `{transform_name}`**")
                report_content.append(T_summary_df.to_markdown(index=False, tablefmt="pipe"))
                report_content.append("\n")
        else:
            report_content.append("\n  _No transformation-specific metrics to display._")
        report_content.append("\n</details>\n") # Blank line after details block

    except Exception as e:
        logger.exception(f"Could not generate summary table: {e}") # Use exception for traceback
        report_content.append("\n_Error generating summary table._\n") # Blank line before error, single after

    # --- Error Rate Distribution Plots --- #
    # Attempt to generate plots. This is done after summary, before detailed results.
    plot_section_content = []
    try:
        # Assuming 'cer' and 'wer' are the metrics you want to plot if available
        # You can customize this list based on available metrics in config or df
        metrics_to_plot_for_dist = []
        if not df.empty and 'metrics' in df.columns:
            sample_metrics = df['metrics'].iloc[0]
            if isinstance(sample_metrics, dict):
                if 'cer' in sample_metrics: metrics_to_plot_for_dist.append('cer')
                if 'wer' in sample_metrics: metrics_to_plot_for_dist.append('wer')
        
        if metrics_to_plot_for_dist: # Only proceed if we have metrics to plot
            plot_files_map = generate_error_distribution_plots(df, output_dir, metrics_to_plot=metrics_to_plot_for_dist)
            if plot_files_map:
                plot_section_content.append("## Error Rate Distributions by Model")
                for plot_key, plot_rel_path in plot_files_map.items():
                    # plot_key could be e.g. "cer_distribution_plot"
                    # Construct a title from the key
                    plot_title = plot_key.replace("_", " ").replace(" distribution plot", "").title()
                    plot_section_content.append(f"### {plot_title} Distribution") # Heading provides spacing
                    plot_section_content.append(f"![{plot_title} Distribution for all models]({plot_rel_path})")
                    plot_section_content.append("\n") # Blank line after image
    except Exception as e:
        logger.error(f"Failed to generate or include error distribution plots: {e}")
        plot_section_content.append("\n_Could not generate error distribution plots._\n") # Blank line before error
    
    if plot_section_content: # Add plot section if content was generated
        report_content.extend(plot_section_content)

    # --- Detailed Results Per Image --- #
    report_content.append(f"## Detailed Results")

    for (pair_idx, original_image), group in df.groupby(['image_pair_index', 'original_image_path']):
        img_idx_for_slug = df['original_image_path'].unique().tolist().index(original_image)
        # current_image_slug = image_slugs[img_idx_for_slug] # Slug no longer needed for header ID
        report_content.append(f"### Source Image: `{os.path.basename(original_image)}`") # Removed custom ID
        gt_path = group['ground_truth_path'].iloc[0]
        report_content.append(f"Ground Truth: `{os.path.basename(gt_path)}`")
        # Maybe display GT text if short?
        # report_content.append("```text")
        # report_content.append(load_ground_truth_text(gt_path))
        # report_content.append("```\n")

        # Group by variant within this image
        for variant_desc, variant_group in group.groupby('variant_desc'):
            report_content.append(f"#### Variant: `{variant_desc}`")

            # Embed the variant image
            variant_image_path_abs = variant_group['variant_image_path'].iloc[0]
            if os.path.exists(variant_image_path_abs):
                # Use relative path for Markdown if report is in output_dir
                variant_image_path_rel = os.path.relpath(variant_image_path_abs, start=output_dir)
                # Use HTML img tag to control height
                report_content.append(f'<img src="{variant_image_path_rel}" alt="{variant_desc}" style="max-height: 300px;"><br>') # <br> ensures line break
            else:
                report_content.append(f"_(Image not found: {variant_image_path_abs})_<br>")

            # Table for this variant's results
            variant_table = []
            # Add Provider and Model columns, add Token Counts
            headers = ["Provider", "Model", "Latency (s)", "Cost / 1k images ($)", "Input Tokens", "Output Tokens", "CER", "WER", "Error"] # Removed Snippet and Diff from table headers
            variant_table.append(headers)
            variant_table.append([":-" for _ in headers]) # Separator line

            all_diff_details_for_variant = [] # Store all diffs for this variant

            for _, row in variant_group.sort_values(by=['provider', 'model_identifier']).iterrows(): # Sort rows for consistency
                error_msg = row.get('error') # Don't default to '', check for None/NaN later
                # error_msg_display = sanitize_for_markdown_table(error_msg) if pd.notna(error_msg) else "" # Not used directly in table anymore, but kept for context if needed elsewhere
                
                extracted_text_raw = row['extracted_text']
                # text_display_raw will be used for diff generation
                text_display_raw = str(extracted_text_raw) if pd.notna(extracted_text_raw) else "" 
                
                gt_for_diff = ""
                try:
                    with open(row['ground_truth_path'], 'r', encoding='utf-8') as f_gt:
                        gt_for_diff = f_gt.read()
                except Exception as e_gt:
                    logger.warning(f"Could not load GT for diff: {row['ground_truth_path']}, error: {e_gt}")
                    gt_for_diff = "(Ground truth not available for diff)"

                diff_details = ""
                # Generate diff for this specific row (model)
                if pd.notna(extracted_text_raw) and gt_for_diff and gt_for_diff != "(Ground truth not available for diff)":
                    diff_output = _generate_text_diff(gt_for_diff, text_display_raw)
                    diff_details = (
                        f"<details><summary>Show Diff for {row['provider']} - `{_sanitize_for_inline_markdown_code(row['model_identifier'])}`</summary>"
                        f"\n\n<b>Ground Truth:</b><pre><code>{gt_for_diff}</code></pre>" # gt_for_diff is raw text, HTML pre/code will handle it.
                        f"\n\n<b>Extracted:</b><pre><code>{text_display_raw}</code></pre>" # text_display_raw is raw text.
                        f"\n\n<b>Diff:</b>\n{diff_output}" 
                        f"</details>"
                    )
                    all_diff_details_for_variant.append(diff_details)
                
                # Prepare error message for inline code display in table
                processed_error_for_inline_code = _sanitize_for_inline_markdown_code(error_msg)
                error_cell_content = f"`{processed_error_for_inline_code}`" if processed_error_for_inline_code else ""
                 
                metrics_data = row.get('metrics', {})
                cer_val = metrics_data.get('cer', float('nan'))
                wer_val = metrics_data.get('wer', float('nan'))

                current_cost = row['cost']
                cost_per_1000 = current_cost * 1000 if pd.notna(current_cost) else current_cost

                model_id_for_table = _sanitize_for_inline_markdown_code(row['model_identifier'])

                variant_table.append([
                    f"{row['provider']}",
                    f"`{model_id_for_table}`",
                    format_metric(row['latency_seconds']),
                    format_metric(cost_per_1000),
                    format_metric(row['input_tokens']),
                    format_metric(row['output_tokens']),
                    format_metric(cer_val),
                    format_metric(wer_val),
                    error_cell_content,
                ])

            # Format the table into Markdown
            for row_idx, row_items in enumerate(variant_table):
                 report_content.append("| " + " | ".join(map(str, row_items + [''] * (len(headers) - len(row_items)) )) + " |")

            # Append all collected diffs for this variant after its table
            if all_diff_details_for_variant:
                report_content.append("\n") # Ensure a newline before the first diff details
                report_content.extend(all_diff_details_for_variant)

            report_content.append("\n\n") # Ensure double newline for paragraph break after table and its diffs

    # Write report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    logger.info(f"Markdown report generated successfully at: {report_path}")