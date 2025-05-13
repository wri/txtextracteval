#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration loading and validation for txtextracteval experiments."""

import yaml
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Define default values or structure if needed
DEFAULT_CONFIG = {
    "methods": [
        {"type": "tesseract"}, # Default to at least run Tesseract
    ],
    "transformations": [],
    "metrics": ["cer", "wer", "latency", "cost"],
    "output": {
        "directory": "./txtextract_results",
        "report_filename": "report.md"
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary representing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file cannot be parsed.
        ValueError: If required configuration keys are missing.
    """
    logger.info(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # empty file case
            config = {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read configuration file {config_path}: {e}")
        raise

    # Basic validation for required top-level keys
    if "images" not in config:
        raise ValueError("Configuration error: 'images' section is required.")
    if "ground_truth" not in config:
        raise ValueError("Configuration error: 'ground_truth' section is required.")

    # Apply defaults for missing optional sections (simple merge)
    # A more robust merge would handle nested defaults better
    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config) # User config overrides defaults
    # Ensure nested output defaults are present if output section exists but is partial
    if 'output' in config and isinstance(config['output'], dict):
        final_config['output'] = {**DEFAULT_CONFIG['output'], **config['output']}

    # Validate image/ground truth list lengths if both are lists
    # Needs refinement based on how directories vs lists vs maps are handled
    if isinstance(final_config.get('images'), list) and isinstance(final_config.get('ground_truth'), list):
        if len(final_config['images']) != len(final_config['ground_truth']):
            logger.warning(
                f"Number of images ({len(final_config['images'])}) does not match "
                f"number of ground truths ({len(final_config['ground_truth'])}). "
                f"Ensure they correspond correctly by index."
            )
            # Consider raising ValueError if strict matching is required

    logger.info("Configuration loaded and validated successfully.")
    return final_config

# Ehhhh how to potentially structure image/gt handling later:
def resolve_image_gt_pairs(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Processes the 'images' and 'ground_truth' sections to create pairs.

    Handles cases where inputs are lists, directories, or maps.
    (This is a placeholder for logic needed in the runner)
    """
    # Placeholder logic?
    image_paths = config['images']
    gt_paths = config['ground_truth']
    pairs = []

    if isinstance(image_paths, list) and isinstance(gt_paths, list):
        if len(image_paths) == len(gt_paths):
            for img, gt in zip(image_paths, gt_paths):
                pairs.append({"image": img, "ground_truth": gt})
        else:
             # Handle mismatch as per warning/error policy
             pass
    # TODO: directory scanning, mapping, single file GT, etc.

    if not pairs:
        logger.warning("Could not resolve image/ground truth pairs from config.")

    return pairs 