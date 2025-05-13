#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for Markdown report generation."""

import pytest
import os
import pandas as pd
from unittest.mock import patch, mock_open

# Mock pandas before import if necessary, but usually it's fine
# Mock datetime to control generated timestamp
from datetime import datetime

# Import the function to test
from txtextracteval.report import generate_markdown_report, format_metric

# --- Fixtures --- #

@pytest.fixture
def sample_results() -> list:
    """Provides a sample list of result dictionaries."""
    return [
        {
            'image_pair_index': 0,
            'original_image_path': 'data/image1.png',
            'ground_truth_path': 'data/gt1.txt',
            'variant_desc': 'original',
            'variant_image_path': 'results/exp1/image1_original.png',
            'method_type': 'tesseract',
            'method_config': {},
            'extracted_text': 'Tess Original',
            'latency_seconds': 0.45678,
            'cost': 0.0,
            'metrics': {'cer': 0.05, 'wer': 0.1},
            'misc': None,
            'error': None
        },
        {
            'image_pair_index': 0,
            'original_image_path': 'data/image1.png',
            'ground_truth_path': 'data/gt1.txt',
            'variant_desc': 'blur_k3',
            'variant_image_path': 'results/exp1/image1_blur_k3.png',
            'method_type': 'tesseract',
            'method_config': {},
            'extracted_text': 'Tess Blur',
            'latency_seconds': 0.5123,
            'cost': 0.0,
            'metrics': {'cer': 0.15, 'wer': 0.25},
            'misc': None,
            'error': None
        },
                {
            'image_pair_index': 0,
            'original_image_path': 'data/image1.png',
            'ground_truth_path': 'data/gt1.txt',
            'variant_desc': 'original',
            'variant_image_path': 'results/exp1/image1_original.png',
            'method_type': 'llm_api',
            'method_config': {'provider': 'gemini'},
            'extracted_text': 'LLM Original',
            'latency_seconds': 4.876,
            'cost': 0.02,
            'metrics': {'cer': 0.02, 'wer': 0.05},
            'misc': {'usage': 'mock'},
            'error': None
        },
        {
            'image_pair_index': 0,
            'original_image_path': 'data/image1.png',
            'ground_truth_path': 'data/gt1.txt',
            'variant_desc': 'blur_k3',
            'variant_image_path': 'results/exp1/image1_blur_k3.png',
            'method_type': 'llm_api',
            'method_config': {'provider': 'gemini'},
            'extracted_text': None, # Simulate failed extraction
            'latency_seconds': 5.123,
            'cost': 0.02,
            'metrics': {},
            'misc': {'usage': 'mock'},
            'error': 'API Timeout'
        },
    ]

@pytest.fixture
def sample_report_config() -> dict:
    """Provides a sample config for context in the report."""
    return {
        "images": ["data/image1.png"],
        "ground_truth": ["data/gt1.txt"],
        "methods": [{"type": "tesseract"}, {"type": "llm_api"}],
        "transformations": [{"name": "blur", "params": {"kernel_size": 3}}],
        "metrics": ["cer", "wer", "latency"],
        "output": {
            "directory": "results/exp1",
            "report_filename": "report.md"
        }
    }

# --- Tests for format_metric --- #

def test_format_metric_float():
    assert format_metric(0.123456) == "0.1235"
    assert format_metric(123.456) == "123.4560"
    assert format_metric(0.0) == "0.0000"
    assert format_metric(1e-5) == "1.00e-05"
    assert format_metric(1e7) == "1.00e+07"

def test_format_metric_int():
    assert format_metric(10) == "10"

def test_format_metric_string():
    assert format_metric("N/A") == "N/A"

def test_format_metric_none():
    assert format_metric(None) == "N/A"

# --- Tests for generate_markdown_report --- #

@patch("builtins.open", new_callable=mock_open)
@patch("txtextracteval.report.os.path.exists", return_value=True) # Assume images exist
@patch("txtextracteval.report.os.path.relpath", lambda p, start: os.path.basename(p)) # Simplify relpath
@patch("txtextracteval.report.datetime")
def test_generate_report_structure(mock_dt, mock_relpath, mock_exists, mock_file_open, sample_results, sample_report_config):
    """Test the overall structure and content of the generated report."""
    # Arrange
    mock_now = datetime(2024, 1, 1, 10, 30, 0)
    mock_dt.datetime.now.return_value = mock_now
    output_dir = sample_report_config['output']['directory']
    filename = sample_report_config['output']['report_filename']
    report_path = os.path.join(output_dir, filename)

    # Act
    generate_markdown_report(sample_results, sample_report_config, output_dir, filename)

    # Assert
    mock_file_open.assert_called_once_with(report_path, 'w', encoding='utf-8')
    # Get all written content by joining mock calls
    handle = mock_file_open()
    written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)

    # Check for key sections and content
    assert "# Text Extraction Evaluation Report" in written_content
    assert "Generated on: 2024-01-01 10:30:00" in written_content
    assert "## Experiment Overview" in written_content
    assert "*   **Methods Tested:** ['tesseract', 'llm_api']" in written_content
    assert "## Summary Metrics (Averages)" in written_content
    assert "| Provider   | Model   |   Avg Latency (s) |   Avg Cost / 1k images ($) |    Avg CER |    Avg WER |" in written_content
    assert "| llm_api    | gemini  |            4.9995 |                    20.0000 |     0.0200 |     0.0500 |" in written_content
    assert "| tesseract  | tesseract |            0.4845 |                     0.0000 |     0.1000 |     0.1750 |" in written_content
    assert "## Detailed Results" in written_content
    assert "### Source Image: `image1.png`" in written_content
    assert "#### Variant: `original`" in written_content
    assert "![original](image1_original.png)" in written_content
    assert "| Provider  | Model     |   Latency (s) |   Cost / 1k images ($) | Extracted Text (Snippet)   |    CER |    WER | Error         |" in written_content
    assert "| tesseract | tesseract |        0.4568 |                 0.0000 | `Tess Original`            | 0.0500 | 0.1000 |               |" in written_content
    assert "| llm_api   | gemini    |        4.8760 |                20.0000 | `LLM Original`             | 0.0200 | 0.0500 |               |" in written_content
    assert "#### Variant: `blur_k3`" in written_content
    assert "![blur_k3](image1_blur_k3.png)" in written_content
    assert "| tesseract | tesseract |        0.5123 |                 0.0000 | `Tess Blur`                | 0.1500 | 0.2500 |               |" in written_content
    assert "| llm_api   | gemini    |        5.1230 |                20.0000 | ``                         | N/A    | N/A    | `API Timeout` |" in written_content

@patch("builtins.open", new_callable=mock_open)
def test_generate_report_no_results(mock_file_open, sample_report_config):
    """Test report generation when the results list is empty."""
    output_dir = sample_report_config['output']['directory']
    filename = sample_report_config['output']['report_filename']
    report_path = os.path.join(output_dir, filename)

    generate_markdown_report([], sample_report_config, output_dir, filename)

    mock_file_open.assert_called_once_with(report_path, 'w', encoding='utf-8')
    handle = mock_file_open()
    written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)

    assert "# Text Extraction Evaluation Report" in written_content
    assert "**Warning:** No results were generated" in written_content
    assert "## Summary Metrics (Averages)" not in written_content # No summary if no results 