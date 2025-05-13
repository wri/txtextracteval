#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""txtextracteval package main entry point and metadata."""

import importlib.metadata
import logging

from .config import load_config
from .runner import run_experiment
# Import from report subpackage
from .report import generate_markdown_report
# Import specific metrics if needed at top level, or let users import from txtextracteval.metrics
from .metrics import calculate_cer, calculate_wer

# Get version from pyproject.toml
try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0" # Default if package not installed

# Configure basic logging for the package
# Consumers of the library can override this
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"txtextracteval version {__version__} loaded.")

# Remove the old placeholder function
# def main() -> None:
#     print("Hello from txtextracteval!")

__all__ = [
    "load_config",
    "run_experiment",
    "generate_markdown_report",
    "calculate_cer",
    "calculate_wer",
    "__version__"
]
