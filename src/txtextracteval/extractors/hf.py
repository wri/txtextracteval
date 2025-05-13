#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hugging Face Transformer based text extractor."""

import logging
from typing import Any, Dict
import numpy as np
from PIL import Image
import cv2
import torch

# Try importing necessary libraries
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from transformers.image_utils import load_image
except ImportError:
    raise ImportError("Please install transformers and torch: `uv add transformers torch sentencepiece pytesseract`")

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Default prompt for the model, can be made configurable if needed
DEFAULT_PROMPT_TEXT = "Extract text from this image verbatim."
# Default max_new_tokens, suitable for SmolVLM-256M-Instruct description tasks
DEFAULT_MAX_NEW_TOKENS = 500

class HFExtractor(BaseExtractor):
    """Extractor using Hugging Face AutoProcessor and AutoModelForVision2Seq."""

    processor: Any = None
    model: Any = None
    model_name_loaded: str | None = None
    loaded_device: str | None = None
    # To track image processor args used for loading, to ensure consistency or allow reloading
    loaded_image_processor_args: Dict[str, Any] | None = None

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the extractor, loading the specified model and processor.

        Args:
            config: Configuration dictionary, expecting 'model' key for model name.
        """
        super().__init__(config)
        self.model_name = self.config.get("model", DEFAULT_MODEL)

        # Device selection: CUDA -> MPS (for M1/M2 Macs) -> CPU
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            default_device = "mps"
        else:
            default_device = "cpu"
        
        self.device = self.config.get("device", default_device)
        logger.info(f"Selected device: {self.device}")

        # Attention implementation
        if self.device == "cuda":
            default_attn_implementation = "flash_attention_2"
        else: 
            default_attn_implementation = "eager"
        self.attn_implementation = self.config.get("attn_implementation", default_attn_implementation)
        logger.info(f"Using attention implementation: {self.attn_implementation}")

        # Torch dtype selection
        if self.device == "cuda":
            default_torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            default_torch_dtype = torch.float16
        else: 
            default_torch_dtype = torch.float32
        self.torch_dtype = self.config.get("torch_dtype", default_torch_dtype)
        logger.info(f"Using torch dtype: {self.torch_dtype}")

        # Image processor arguments (e.g., for size)
        # Default N=4, patch_size=512 -> longest_edge = 2048
        default_image_processor_args = {"size": {"longest_edge": 4 * 512}}
        self.image_processor_args = self.config.get("image_processor_args", default_image_processor_args)
        logger.info(f"Using image_processor_args: {self.image_processor_args}")

        self.quantization_config_dict = self.config.get("quantization_config", None)
        if self.quantization_config_dict and not (self.device == "cuda"):
            logger.warning("Quantization config provided but not on CUDA. BitsAndBytes quantization might not apply or work as expected.")
        
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        """Loads the Hugging Face processor and model."""
        # Reload if model/device/image_processor_args change
        if HFExtractor.model is None or \
           HFExtractor.processor is None or \
           HFExtractor.model_name_loaded != self.model_name or \
           HFExtractor.loaded_device != self.device or \
           HFExtractor.loaded_image_processor_args != self.image_processor_args: # Check image_proc_args
            
            logger.info(f"Loading Hugging Face model ({self.model_name}) and processor...")
            logger.info(f"  Device: {self.device}, Dtype: {self.torch_dtype}")
            logger.info(f"  Image Processor Args: {self.image_processor_args}")
            logger.info(f"  Attention Implementation: {self.attn_implementation}")

            try:
                quant_config_obj = None
                if self.quantization_config_dict and self.device == "cuda":
                    logger.info(f"  Applying BitsAndBytes quantization: {self.quantization_config_dict}")
                    quant_config_obj = BitsAndBytesConfig(**self.quantization_config_dict)
                elif self.quantization_config_dict:
                     logger.warning(f"  Quantization config {self.quantization_config_dict} ignored (device is not CUDA).")

                # Pass image_processor_args to AutoProcessor
                # These are typically forwarded to the underlying image processor's from_pretrained method
                HFExtractor.processor = AutoProcessor.from_pretrained(self.model_name, **self.image_processor_args)
                
                model_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "_attn_implementation": self.attn_implementation
                }
                if quant_config_obj:
                    model_kwargs["quantization_config"] = quant_config_obj
                
                HFExtractor.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    **model_kwargs
                ).to(self.device)
                
                HFExtractor.model_name_loaded = self.model_name
                HFExtractor.loaded_device = self.device
                HFExtractor.loaded_image_processor_args = self.image_processor_args # Store args used
                logger.info(f"Model and processor loaded successfully for {self.model_name} on {self.device}.")
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model/processor for {self.model_name}: {e}")
                logger.exception("Full traceback for model/processor loading error:")
                raise RuntimeError(f"Failed to load HF model/processor {self.model_name}") from e
        else:
            logger.info(f"Reusing existing model ({HFExtractor.model_name_loaded} on {HFExtractor.loaded_device} with image_proc_args {HFExtractor.loaded_image_processor_args}) and processor.")

    def extract(self, image: np.ndarray | str) -> ExtractionResult:
        """
        Extracts text using the loaded Hugging Face model and processor.

        Args:
            image: Image input. Can be a path (str) or a NumPy array (OpenCV BGR format).

        Returns:
            An ExtractionResult containing the extracted text.
        """
        if self.processor is None or self.model is None:
            # This case should ideally be prevented by constructor calling _load_model_and_processor
            logger.error("Hugging Face processor or model is not loaded.")
            raise RuntimeError("Hugging Face processor or model is not loaded.")

        try:
            if isinstance(image, str):
                img_input_pil = load_image(image) # Returns PIL image
                logger.debug(f"Loaded image from path: {image}. Size: {img_input_pil.size}, Mode: {img_input_pil.mode}")
            elif isinstance(image, np.ndarray):
                img_input_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                logger.debug(f"Using provided NumPy array as image input. Converted to PIL. Size: {img_input_pil.size}, Mode: {img_input_pil.mode}")
            else:
                raise TypeError("Unsupported image type. Provide path (str) or NumPy array (BGR).")

            # Construct messages payload as per SmolDocling example
            # The prompt can be made configurable if needed.
            user_prompt_text = self.config.get("prompt_text", DEFAULT_PROMPT_TEXT)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt_text}
                    ]
                },
            ]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[img_input_pil], return_tensors="pt").to(self.device)
            
            if 'pixel_values' in inputs:
                 logger.debug(f"Processed image (pixel_values) shape: {inputs['pixel_values'].shape}")
            else:
                 logger.warning("`pixel_values` not found in processor output. Image might not have been processed correctly.")

            # Generation parameters - can be made configurable
            max_new_tokens = self.config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
            # For verbatim extraction, greedy decoding is often best.
            # do_sample=False is default for generate if not specified.
            # temperature=0.0 (or very low) for greedy if do_sample=True.
            # For now, relying on generate's defaults which should be greedy.
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # Decode the generated IDs.
            # The example uses batch_decode and then lstrip() on the first result.
            # skip_special_tokens=True might simplify this. Test to see.
            # For now, following example's spirit but aiming for clean text.
            # The example processes "doctags", we want the direct text.
            # The output structure of `generated_ids` relative to `inputs.input_ids` needs care for decoding.
            # Usually, we need to decode only the newly generated part.
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]

            # Using skip_special_tokens=True to get cleaner text directly.
            decoded_texts = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=True # Changed from False in example to get cleaner text
            )
            extracted_text = decoded_texts[0].strip()

            # SmolVLM models often prefix responses with "Assistant: "
            assistant_prefix = "Assistant: "
            if extracted_text.startswith(assistant_prefix):
                extracted_text = extracted_text[len(assistant_prefix):].strip()
            
            logger.debug(f"HF model {self.model_name_loaded} extracted text (first 100 chars): '{extracted_text[:100]}...'")
            return ExtractionResult(text=extracted_text, latency_seconds=-1)

        except FileNotFoundError: # Specifically for image path
            logger.error(f"Image file not found at path: {str(image)}")
            raise
        except Exception as e:
            logger.error(f"Error during Hugging Face model inference for {self.model_name_loaded}: {e}")
            # Log the full traceback for better debugging
            logger.exception("Full traceback for HF inference error:")
            raise RuntimeError(f"Hugging Face inference failed for {self.model_name_loaded}") from e 