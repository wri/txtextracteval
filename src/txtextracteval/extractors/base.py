#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base class for text extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class ExtractionResult(NamedTuple):
    """Structured result from an extraction method."""
    text: str
    latency_seconds: float
    cost: float = 0.0 # Cost (e.g., API cost), default to 0 for local models
    misc: Dict[str, Any] | None = None # For additional metadata (e.g., tokens used)

class BaseExtractor(ABC):
    """Abstract Base Class for all text extraction methods."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the extractor, potentially with method-specific config."""
        self.config = config if config else {}
        logger.info(f"Initializing extractor: {self.__class__.__name__}")

    @abstractmethod
    def extract(self, image: np.ndarray | str) -> ExtractionResult:
        """
        Extracts text from the given image.

        Args:
            image: Image input. Can be a NumPy array (loaded by OpenCV) or a file path.
                   Implementations should handle whichever format they expect.

        Returns:
            An ExtractionResult object containing text, latency, cost, and optional metadata.

        Raises:
            Exceptions on failure (e.g., model loading error, API error, file not found).
        """
        pass

    def run_extraction(self, image: np.ndarray | str) -> ExtractionResult:
        """
        Wrapper method to time the extraction process and handle basic logging.
        Subclasses should implement the core logic in the `extract` method.
        """
        start_time = time.perf_counter()
        try:
            result = self.extract(image)
            end_time = time.perf_counter()
            # Ensure latency is included, overriding if necessary
            final_result = ExtractionResult(
                text=result.text,
                latency_seconds=end_time - start_time, # Use measured time
                cost=result.cost,
                misc=result.misc
            )
            logger.info(f"{self.__class__.__name__} extraction finished in {final_result.latency_seconds:.4f} seconds.")
            return final_result
        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time
            logger.error(f"{self.__class__.__name__} extraction failed after {latency:.4f} seconds: {e}")
            # Re-raise the exception so the runner knows about the failure
            raise 