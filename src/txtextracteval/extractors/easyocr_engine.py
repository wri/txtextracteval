#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text extractor using EasyOCR."""

import logging
import time
from typing import List, Dict, Any
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

class EasyOCRExtractor(BaseExtractor):
    """Extractor using the EasyOCR library."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the EasyOCR extractor.

        Args:
            config: Configuration dictionary. Expected keys:
                    'languages': List of language codes (e.g., ['en', 'ch_sim']). Defaults to ['en'].
                    'gpu': Boolean, whether to use GPU. Defaults to True if available.
                    Other EasyOCR Reader parameters can also be passed.
        """
        super().__init__(config)
        if not EASYOCR_AVAILABLE:
            logger.error("EasyOCR library not found. Please install it to use EasyOCRExtractor.")
            raise ImportError("EasyOCR library not found. pip install easyocr")

        self.languages = self.config.get('languages', ['en'])
        self.gpu = self.config.get('gpu', True)
        # Pass through other config params to EasyOCR Reader if needed
        reader_params = {k: v for k, v in self.config.items() if k not in ['languages', 'gpu']}

        try:
            logger.info(f"Initializing EasyOCR Reader with languages: {self.languages}, GPU: {self.gpu}")
            # EasyOCR might print to stdout during model download, which is fine.
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, **reader_params)
            logger.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR Reader: {e}")
            raise RuntimeError(f"EasyOCR Reader initialization failed: {e}") from e

    def extract(self, image_input: np.ndarray | str) -> ExtractionResult:
        """Extracts text from an image using EasyOCR.

        Args:
            image_input: NumPy array (BGR format) or path to an image file.

        Returns:
            ExtractionResult containing the extracted text, latency, and cost (0.0).
        """
        start_time = time.perf_counter()
        extracted_text = ""
        misc_data = {}

        try:
            # EasyOCR's readtext expects a file path or a NumPy array (it handles BGR/RGB internally).
            # It returns a list of (bbox, text, prob) tuples if detail=1, or just list of text if detail=0.
            # We want paragraph=True for better text flow, detail=0 to get just text strings.
            results = self.reader.readtext(image_input, detail=0, paragraph=True)
            extracted_text = "\n".join(results) # Join paragraphs/text blocks with newlines
            
            logger.debug(f"EasyOCR extracted text (first 100 chars): '{extracted_text[:100]}...'")
        except Exception as e:
            logger.error(f"Error during EasyOCR text extraction: {e}")
            # Store error in misc_data or re-raise as specific error type?
            # For now, conform to how other extractors might return on failure (empty text, error in result)
            # ExtractionResult expects text, latency, cost, misc. Error is handled by run_extraction.
            # So, we let the exception propagate to be caught by `run_extraction` in `runner.py`.
            raise RuntimeError(f"EasyOCR extraction failed: {e}") from e
        finally:
            latency_seconds = time.perf_counter() - start_time
            logger.info(f"EasyOCR extraction took {latency_seconds:.4f} seconds.")

        return ExtractionResult(
            text=extracted_text,
            latency_seconds=latency_seconds,
            cost=0.0,  # EasyOCR is local, so cost is 0
            misc=misc_data
        ) 