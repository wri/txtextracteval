import pytesseract
from PIL import Image
import logging
import numpy as np
import cv2 # Needed for potential NumPy conversion
from typing import Any, Dict

from .base import BaseExtractor, ExtractionResult

# Configure logging
logger = logging.getLogger(__name__)

class TesseractExtractor(BaseExtractor):
    """Extractor using Tesseract OCR."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize Tesseract extractor.

        Args:
            config: Configuration dictionary. Can contain tesseract options
                    like 'lang' or 'psm'.
        """
        super().__init__(config)
        # Store Tesseract specific configs if needed
        self.lang = self.config.get("lang") # Example: 'eng', 'fra', etc.
        self.psm = self.config.get("psm") # Example: '3', '6', etc.
        logger.info(f"Tesseract configured with lang={self.lang}, psm={self.psm}")

    def extract(self, image: np.ndarray | str) -> ExtractionResult:
        """
        Extracts text from an image file or NumPy array using Tesseract OCR.

        Args:
            image: Path to the image file (str) or image as NumPy array (BGR).

        Returns:
            An ExtractionResult object.
        """
        img_input: Image.Image
        image_source_for_log: str

        if isinstance(image, str):
            image_source_for_log = image
            try:
                # Pytesseract works well with PIL Images
                img_input = Image.open(image)
            except FileNotFoundError:
                logger.error(f"Error: Image file not found at {image_source_for_log}")
                raise
        elif isinstance(image, np.ndarray):
            image_source_for_log = "NumPy array"
            # Convert BGR NumPy array to RGB PIL Image
            img_input = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError("Unsupported image type for Tesseract. Provide path (str) or NumPy array.")

        try:
            # Construct tesseract config string
            custom_config = ""
            if self.lang:
                custom_config += f"-l {self.lang} "
            if self.psm:
                custom_config += f"--psm {self.psm}"
            custom_config = custom_config.strip()

            # Use pytesseract to do OCR on the image
            text = pytesseract.image_to_string(img_input, config=custom_config)
            logger.info(f"Tesseract successfully extracted text from {image_source_for_log} (config='{custom_config}')")
            extracted_text = text.strip()

            # Latency calculated by wrapper, cost is 0
            return ExtractionResult(text=extracted_text, latency_seconds=-1, cost=0.0)

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH.")
            raise # Re-raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Tesseract OCR on {image_source_for_log}: {e}")
            raise # Re-raise 