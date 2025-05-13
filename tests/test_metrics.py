#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the metrics calculation functions."""

import pytest
import math
from txtextracteval.metrics import calculate_cer, calculate_wer

# --- Tests for calculate_cer --- #

def test_cer_perfect_match():
    """Test CER when ground truth and extracted text match perfectly."""
    assert calculate_cer("hello world", "hello world") == 0.0

def test_cer_complete_mismatch():
    """Test CER with completely different strings of the same length."""
    # distance = 5, len = 5 -> CER = 1.0
    assert calculate_cer("abcde", "fghij") == 1.0

def test_cer_insertion():
    """Test CER with an insertion."""
    # distance = 2 (insert 'x', insert ' '), len = 11 -> CER = 2/11
    assert calculate_cer("hello world", "hello x world") == pytest.approx(2/11)

def test_cer_deletion():
    """Test CER with a deletion."""
    # distance = 1, len = 11 -> CER = 1/11
    assert calculate_cer("hello world", "helloworld") == pytest.approx(1/11)

def test_cer_substitution():
    """Test CER with a substitution."""
    # distance = 1, len = 11 -> CER = 1/11
    assert calculate_cer("hello world", "hello vorld") == pytest.approx(1/11)

def test_cer_empty_strings():
    """Test CER with empty strings."""
    assert calculate_cer("", "") == 0.0

def test_cer_empty_ground_truth():
    """Test CER when ground truth is empty but extracted is not."""
    assert calculate_cer("", "extracted text") == 1.0

def test_cer_empty_extracted():
    """Test CER when extracted is empty but ground truth is not."""
    # distance = 11, len = 11 -> CER = 1.0
    assert calculate_cer("ground truth", "") == 1.0

def test_cer_with_whitespace():
    """Test CER ignores leading/trailing whitespace."""
    assert calculate_cer("  hello world\n", "hello world  ") == 0.0

def test_cer_case_sensitive():
    """Test CER is case-sensitive."""
    # distance = 1, len = 5 -> CER = 0.2
    assert calculate_cer("Hello", "hello") == pytest.approx(0.2)

def test_cer_longer_extraction():
    """Test CER can be > 1 if extraction is much longer."""
    # distance = 10 (insert " plus more"), len = 5 -> CER = 10/5 = 2.0
    assert calculate_cer("short", "short plus more") == pytest.approx(2.0)

def test_cer_non_string_input():
    """Test CER handles non-string input gracefully (returns NaN)."""
    assert math.isnan(calculate_cer(123, "string"))
    assert math.isnan(calculate_cer("string", None))
    assert math.isnan(calculate_cer(None, None))

# --- Tests for calculate_wer --- #

def test_wer_perfect_match():
    """Test WER when ground truth and extracted text match perfectly."""
    assert calculate_wer("hello world", "hello world") == 0.0

def test_wer_complete_mismatch():
    """Test WER with completely different words."""
    # 2 substitutions, 2 words -> WER = 1.0
    assert calculate_wer("one two", "three four") == 1.0

def test_wer_insertion():
    """Test WER with an inserted word."""
    # 1 insertion, 2 words -> WER = 0.5
    assert calculate_wer("hello world", "hello new world") == 0.5

def test_wer_deletion():
    """Test WER with a deleted word."""
    # 1 deletion, 3 words -> WER = 1/3
    assert calculate_wer("hello cruel world", "hello world") == pytest.approx(1/3)

def test_wer_substitution():
    """Test WER with a substituted word."""
    # 1 substitution, 2 words -> WER = 0.5
    assert calculate_wer("hello world", "hello there") == 0.5

def test_wer_empty_strings():
    """Test WER with empty strings."""
    assert calculate_wer("", "") == 0.0

def test_wer_empty_ground_truth():
    """Test WER when ground truth is empty but extracted is not."""
    assert calculate_wer("", "extracted text") == 1.0

def test_wer_empty_extracted():
    """Test WER when extracted is empty but ground truth is not."""
    # distance = 2 words, len = 2 words -> WER = 1.0
    assert calculate_wer("ground truth", "") == 1.0

def test_wer_with_whitespace():
    """Test WER ignores leading/trailing/extra internal whitespace."""
    assert calculate_wer("  hello   world\n", "hello world  ") == 0.0

def test_wer_case_sensitive():
    """Test WER is case-sensitive (via word comparison)."""
    # "Hello" != "hello" -> 1 substitution, 1 word -> WER = 1.0
    assert calculate_wer("Hello", "hello") == 1.0

def test_wer_longer_extraction():
    """Test WER when extraction is longer."""
    # 2 insertions, 2 words -> WER = 1.0
    assert calculate_wer("one two", "one two three four") == 1.0

def test_wer_shorter_extraction():
    """Test WER when extraction is shorter."""
    # 2 deletions, 4 words -> WER = 0.5
    assert calculate_wer("one two three four", "one two") == 0.5

def test_wer_non_string_input():
    """Test WER handles non-string input gracefully (returns NaN)."""
    assert math.isnan(calculate_wer(123, "string"))
    assert math.isnan(calculate_wer("string", None))
    assert math.isnan(calculate_wer(None, None))

# TODO: Add tests for calculate_wer 