#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core pipeline runner for txtextracteval experiments."""

import logging
import os
import cv2
import numpy as np
from typing import Dict, Any, List
import uuid # For unique variant filenames
import time

from .config import load_config, DEFAULT_CONFIG
from .extractors import BaseExtractor, TesseractExtractor, HFExtractor, LLMExtractor, ExtractionResult
from .extractors.easyocr_engine import EasyOCRExtractor # Import EasyOCR
from .transforms import apply_gaussian_blur, adjust_brightness, rotate_image, add_noise, crop_border, apply_skew, apply_ocr_prep_pipeline, reduce_resolution # Import available transforms
from .metrics import calculate_cer, calculate_wer

logger = logging.getLogger(__name__)

# --- Mapping from config names to actual implementations --- #

# Map transform names (from config) to functions
TRANSFORM_REGISTRY = {
    "blur": apply_gaussian_blur,
    "brightness": adjust_brightness,
    "rotate": rotate_image,
    "noise": add_noise,
    "crop": crop_border,
    "skew": apply_skew,
    "ocr_prep": apply_ocr_prep_pipeline,
    "reduce_resolution": reduce_resolution,
    # Add other transforms here as they are implemented
}

# Map method types (from config) to extractor classes
EXTRACTOR_REGISTRY = {
    "tesseract": TesseractExtractor,
    "hf_ocr": HFExtractor,
    "llm_api": LLMExtractor,
    "easyocr": EasyOCRExtractor,
    # Add other extractors here
}
# --------------------------------------------------------- #

def _resolve_image_gt_pairs(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Processes 'images' and 'ground_truth' to create pairs.

    Current simple implementation: Assumes lists of the same length,
    or a single image/gt file path.
    TODO: Add support for directories, mapping files.
    """
    image_input = config['images']
    gt_input = config['ground_truth']
    pairs = []

    if isinstance(image_input, str) and isinstance(gt_input, str):
        # Single image and single ground truth file
        pairs.append({"image": image_input, "ground_truth": gt_input})
    elif isinstance(image_input, list) and isinstance(gt_input, list):
        if len(image_input) != len(gt_input):
            raise ValueError(
                f"Config error: Mismatched number of images ({len(image_input)}) "
                f"and ground truths ({len(gt_input)}). Provide matching lists."
            )
        for img, gt in zip(image_input, gt_input):
            pairs.append({"image": img, "ground_truth": gt})
    # TODO: Add handling for directory input for images + corresponding GT logic
    else:
        raise ValueError("Unsupported format for 'images' or 'ground_truth' in config. Use single path strings or lists of paths.")

    if not pairs:
         raise ValueError("Could not resolve any image/ground truth pairs from config.")

    # Validate file existence
    for pair in pairs:
        if not os.path.exists(pair["image"]):
             raise FileNotFoundError(f"Image file not found: {pair['image']}")
        if not os.path.exists(pair["ground_truth"]):
             raise FileNotFoundError(f"Ground truth file not found: {pair['ground_truth']}")

    logger.info(f"Resolved {len(pairs)} image/ground truth pair(s).")
    return pairs

def _load_ground_truth(gt_path: str) -> str:
    """Loads ground truth text from a file."""
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load ground truth file {gt_path}: {e}")
        raise

def _apply_transformations(image: np.ndarray, transformations: List[Dict[str, Any]], output_dir: str, base_filename: str) -> Dict[str, str | np.ndarray]:
    """Applies configured transformations to an image and saves variants.

    Args:
        image: Original image as NumPy array.
        transformations: List of transformation configs from the main config.
        output_dir: Directory to save transformed images.
        base_filename: Original base filename (without extension) for naming variants.

    Returns:
        Dictionary mapping transformation description (e.g., 'original', 'blur_k3')
        to the NumPy array of the transformed image OR its saved path.
        (Returning path might be better for memory with many variants).
        Let's save and return the path.
    """
    variants = {}
    # Always include the original image
    original_filename = f"{base_filename}_original.png" 
    original_path = os.path.join(output_dir, original_filename)
    try:
        cv2.imwrite(original_path, image)
        variants["original"] = original_path
        logger.debug(f"Saved original image variant to {original_path}")
    except Exception as e:
        logger.error(f"Failed to save original image variant: {e}")
        # Continue without original if save fails?

    current_image = image.copy()

    for transform_config in transformations:
        name = transform_config.get("name")
        params = transform_config.get("params", {})

        if name not in TRANSFORM_REGISTRY:
            logger.warning(f"Unknown transformation '{name}' specified in config. Skipping.")
            continue

        transform_func = TRANSFORM_REGISTRY[name]
        logger.info(f"Applying transformation: {name} with params: {params}")

        try:
            # Apply the transformation function
            transformed_image = transform_func(current_image, **params)

            # Create a descriptive name and save the variant
            param_str = "_".join(f"{k}{v}" for k, v in params.items()) if params else ""
            variant_desc = f"{name}_{param_str}" if param_str else name
            variant_filename = f"{base_filename}_{variant_desc}.png"
            variant_path = os.path.join(output_dir, variant_filename)

            cv2.imwrite(variant_path, transformed_image)
            variants[variant_desc] = variant_path
            logger.debug(f"Saved {variant_desc} image variant to {variant_path}")

        except Exception as e:
            logger.error(f"Failed to apply or save transformation '{name}': {e}")
            # Continue with next transformation

    return variants

def run_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs the full text extraction evaluation experiment based on the config.

    Args:
        config: The loaded configuration dictionary.

    Returns:
        A list of dictionaries, where each dictionary contains the results
        for one method applied to one image variant.
        Example entry:
        {
            'image_pair_index': 0,
            'original_image_path': 'data/image1.png',
            'ground_truth_path': 'data/image1.txt',
            'variant_desc': 'blur_k3',
            'variant_image_path': 'results/exp1/image1_blur_k3.png',
            'method_type': 'tesseract',
            'method_config': { 'lang': 'eng', 'psm': 3 },
            'extracted_text': 'The extracted text...',
            'latency_seconds': 0.5,
            'cost': 0.0,
            'metrics': {
                'cer': 0.05,
                'wer': 0.10
            },
            'misc': None, # Optional metadata from extractor
            'error': None # or error message if extraction failed
        }
    """
    all_results = []
    output_dir = config['output']['directory']
    metrics_to_calculate = config.get('metrics', DEFAULT_CONFIG['metrics'])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    image_gt_pairs = _resolve_image_gt_pairs(config)

    for idx, pair in enumerate(image_gt_pairs):
        original_image_path = pair["image"]
        gt_path = pair["ground_truth"]
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        logger.info(f"Processing pair {idx+1}/{len(image_gt_pairs)}: Image='{original_image_path}', GT='{gt_path}'")

        # Load original image and ground truth
        try:
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                raise ValueError(f"Could not load image file: {original_image_path}")
            ground_truth_text = _load_ground_truth(gt_path)
        except Exception as e:
            logger.error(f"Skipping pair {idx+1} due to error loading image or GT: {e}")
            continue

        # Apply transformations and get paths to variants
        image_variants = _apply_transformations(
            original_image,
            config.get("transformations", []),
            output_dir,
            base_filename
        )

        # Run configured methods on each variant
        for method_config in config.get("methods", []):
            method_type = method_config.get("type")
            specific_config = method_config.get("config", {})

            if method_type not in EXTRACTOR_REGISTRY:
                logger.warning(f"Unknown method type '{method_type}' in config. Skipping.")
                continue

            ExtractorClass = EXTRACTOR_REGISTRY[method_type]

            try:
                # Initialize extractor (potentially loads models)
                extractor = ExtractorClass(config=specific_config)
                logger.info(f"Initialized extractor: {method_type}")
            except Exception as e:
                logger.error(f"Failed to initialize extractor '{method_type}': {e}. Skipping method.")
                # Record failure for all variants for this method?
                # Or just log and continue?
                continue # Skip this method entirely for this image pair

            for variant_desc, variant_image_path in image_variants.items():
                logger.info(f"Running method '{method_type}' on variant '{variant_desc}' ({variant_image_path})")

                result_entry = {
                    'image_pair_index': idx,
                    'original_image_path': original_image_path,
                    'ground_truth_path': gt_path,
                    'variant_desc': variant_desc,
                    'variant_image_path': variant_image_path,
                    'method_type': method_type,
                    'method_config': specific_config,
                    'extracted_text': None,
                    'latency_seconds': None,
                    'cost': None,
                    'metrics': {},
                    'misc': None,
                    'error': None
                }

                try:
                    # Run extraction (handles timing and internal errors)
                    # Pass the image path, extractor should handle loading if needed
                    extraction_result = extractor.run_extraction(variant_image_path)

                    result_entry['extracted_text'] = extraction_result.text
                    result_entry['latency_seconds'] = extraction_result.latency_seconds
                    result_entry['cost'] = extraction_result.cost
                    result_entry['misc'] = extraction_result.misc

                    # Calculate requested metrics
                    if 'cer' in metrics_to_calculate:
                        result_entry['metrics']['cer'] = calculate_cer(ground_truth_text, extraction_result.text)
                    if 'wer' in metrics_to_calculate:
                         result_entry['metrics']['wer'] = calculate_wer(ground_truth_text, extraction_result.text)
                    # Latency and cost are already in the main dict

                except Exception as e:
                    # Catch errors from run_extraction or metric calculation
                    logger.error(f"Error processing variant '{variant_desc}' with method '{method_type}': {e}")
                    result_entry['error'] = str(e)
                    # Latency might be available from run_extraction even if it failed
                    # if isinstance(e, BaseException) and hasattr(e, 'latency'): ? No, run_extraction re-raises

                all_results.append(result_entry)
                # Small delay perhaps? Or rely on API rate limits / model loading time
                # time.sleep(0.1)

    logger.info(f"Experiment finished. Collected {len(all_results)} results.")
    return all_results
