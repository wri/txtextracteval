#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accuracy metrics (CER, WER) based on Levenshtein distance."""

import logging

# Try importing necessary library
try:
    import Levenshtein
except ImportError:
    raise ImportError("Please install python-Levenshtein: `uv add python-Levenshtein`")

logger = logging.getLogger(__name__)

def calculate_cer(ground_truth: str, extracted: str) -> float:
    """Calculates the Character Error Rate (CER).

    CER = (Insertions + Deletions + Substitutions) / Total Characters in Ground Truth

    Args:
        ground_truth: The reference text string.
        extracted: The text string extracted by the OCR/LLM method.

    Returns:
        The Character Error Rate as a float (0.0 to 1.0+).
        Returns 0.0 if ground_truth is empty.
        Can be > 1.0 if extracted text is much longer than ground truth.
    """
    if not isinstance(ground_truth, str) or not isinstance(extracted, str):
        logger.error("Both ground_truth and extracted must be strings.")
        # Or raise TypeError? Returning NaN or a specific value might be better.
        return float('nan')

    gt_clean = ground_truth.strip()
    ext_clean = extracted.strip()

    if not gt_clean:
        # If ground truth is empty, CER is 0 if extracted is also empty,
        # otherwise it's effectively infinite errors (or 1.0 if extracted has content).
        # Let's define CER as 0 only if both are empty.
        return 0.0 if not ext_clean else 1.0

    distance = Levenshtein.distance(gt_clean, ext_clean)
    cer = distance / len(gt_clean)
    logger.debug(f"Calculated CER: distance={distance}, gt_len={len(gt_clean)}, cer={cer:.4f}")
    return cer

def calculate_wer(ground_truth: str, extracted: str) -> float:
    """Calculates the Word Error Rate (WER).

    WER = (Word Insertions + Deletions + Substitutions) / Total Words in Ground Truth

    Args:
        ground_truth: The reference text string.
        extracted: The text string extracted by the OCR/LLM method.

    Returns:
        The Word Error Rate as a float (0.0 to 1.0+).
        Returns 0.0 if ground_truth is empty.
        Can be > 1.0 if extracted text is much longer than ground truth.
    """
    if not isinstance(ground_truth, str) or not isinstance(extracted, str):
        logger.error("Both ground_truth and extracted must be strings.")
        return float('nan')

    # Simple whitespace splitting for words
    # TODO: Consider more sophisticated tokenization (punctuation handling?)
    gt_words = ground_truth.strip().split()
    ext_words = extracted.strip().split()

    if not gt_words:
        # Similar logic to CER for empty ground truth
        return 0.0 if not ext_words else 1.0

    # Calculate Levenshtein distance on the word lists
    distance = Levenshtein.distance(" ".join(gt_words), " ".join(ext_words)) # Note: Levenshtein.distance operates on strings
    # A more direct word-level edit distance might use a different approach if available,
    # but Levenshtein on space-separated words is a common proxy.
    # Alternatively, implement word-level edit distance directly.
    # Let's use the definition based on sequence editing:
    edit_ops = Levenshtein.editops(gt_words, ext_words)
    word_distance = len(edit_ops) # Number of edits (insert, delete, replace) at word level

    wer = word_distance / len(gt_words)
    logger.debug(f"Calculated WER: word_distance={word_distance}, gt_words={len(gt_words)}, wer={wer:.4f}")
    return wer 