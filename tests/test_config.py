#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for configuration loading and validation."""

import pytest
import yaml
import os
from txtextracteval.config import load_config, DEFAULT_CONFIG

# Helper function to create temporary YAML files
@pytest.fixture
def create_yaml_file(tmp_path):
    files_created = []
    def _create_yaml(filename: str, content: dict):
        filepath = tmp_path / filename
        with open(filepath, 'w') as f:
            yaml.dump(content, f)
        files_created.append(filepath)
        return filepath
    yield _create_yaml
    # Cleanup: Optional, pytest tmp_path fixture handles cleanup
    # for f in files_created: os.remove(f)

# --- Tests for load_config --- #

def test_load_valid_config(create_yaml_file):
    """Test loading a basic valid configuration file."""
    valid_content = {
        "images": ["img1.png"],
        "ground_truth": ["gt1.txt"],
        "methods": [{"type": "tesseract"}],
        "output": {"directory": "custom_results"}
    }
    filepath = create_yaml_file("valid.yaml", valid_content)
    config = load_config(str(filepath))

    assert config["images"] == ["img1.png"]
    assert config["ground_truth"] == ["gt1.txt"]
    assert config["methods"] == [{"type": "tesseract"}]
    assert config["output"]["directory"] == "custom_results"
    # Check default applied
    assert config["output"]["report_filename"] == DEFAULT_CONFIG["output"]["report_filename"]
    assert config["metrics"] == DEFAULT_CONFIG["metrics"]

def test_load_config_applies_defaults(create_yaml_file):
    """Test that defaults are applied for missing optional sections."""
    minimal_content = {
        "images": ["img.jpg"],
        "ground_truth": ["gt.txt"],
    }
    filepath = create_yaml_file("minimal.yaml", minimal_content)
    config = load_config(str(filepath))

    assert config["methods"] == DEFAULT_CONFIG["methods"]
    assert config["transformations"] == DEFAULT_CONFIG["transformations"]
    assert config["metrics"] == DEFAULT_CONFIG["metrics"]
    assert config["output"] == DEFAULT_CONFIG["output"]

def test_load_config_overrides_nested_defaults(create_yaml_file):
    """Test that user config overrides nested defaults correctly."""
    partial_output_content = {
        "images": ["img.jpg"],
        "ground_truth": ["gt.txt"],
        "output": {"report_filename": "special_report.md"}
    }
    filepath = create_yaml_file("partial_output.yaml", partial_output_content)
    config = load_config(str(filepath))

    assert config["output"]["report_filename"] == "special_report.md"
    # Default directory should still be applied
    assert config["output"]["directory"] == DEFAULT_CONFIG["output"]["directory"]

def test_load_config_missing_required_images(create_yaml_file):
    """Test error when 'images' section is missing."""
    invalid_content = {"ground_truth": ["gt.txt"]}
    filepath = create_yaml_file("missing_images.yaml", invalid_content)
    with pytest.raises(ValueError, match="'images' section is required"):
        load_config(str(filepath))

def test_load_config_missing_required_gt(create_yaml_file):
    """Test error when 'ground_truth' section is missing."""
    invalid_content = {"images": ["img.png"]}
    filepath = create_yaml_file("missing_gt.yaml", invalid_content)
    with pytest.raises(ValueError, match="'ground_truth' section is required"):
        load_config(str(filepath))

def test_load_config_file_not_found():
    """Test error when the config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

def test_load_config_invalid_yaml(create_yaml_file):
    """Test error when the config file has invalid YAML syntax."""
    filepath = create_yaml_file("invalid_syntax.yaml", {})
    # Manually write invalid YAML
    with open(filepath, 'w') as f:
        f.write("key: value: another_value\n: colon_at_start")

    with pytest.raises(yaml.YAMLError):
        load_config(str(filepath))

def test_load_config_empty_file(create_yaml_file):
    """Test loading an empty YAML file (should raise missing required keys error)."""
    filepath = create_yaml_file("empty.yaml", {})
    # Check it raises because required keys are missing after defaults applied
    with pytest.raises(ValueError, match="'images' section is required"):
        load_config(str(filepath))

# Note: The validation for list length mismatch currently only logs a warning.
# If it were changed to raise an error, a test case should be added here.
# def test_load_config_mismatched_lists(create_yaml_file):
#     content = {
#         "images": ["img1.png", "img2.png"],
#         "ground_truth": ["gt1.txt"]
#     }
#     filepath = create_yaml_file("mismatch.yaml", content)
#     with pytest.raises(ValueError): # Or check log warning
#         load_config(str(filepath)) 