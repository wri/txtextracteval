#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM API based text extractor (Gemini, Ollama, Anthropic, OpenAI)."""

import logging
import os
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import io
import cv2
import base64
import mimetypes

logger = logging.getLogger(__name__) # Move logger up for use in import block

# --- Library Imports ---

# Granular check for Google GenAI components because this was a pain to get working
_genai_imported = False
_types_imported = False
_api_error_imported = False
genai = None
types = None
GoogleAPIError = Exception # Default placeholder

try:
    from google import genai
    _genai_imported = True
except ImportError as e:
    logger.warning(f"Failed to import 'google.genai': {e}. Check google-genai installation.")

try:
    from google.genai import types
    _types_imported = True
except ImportError as e:
    logger.warning(f"Failed to import 'google.genai.types': {e}. Check google-genai installation.")
except AttributeError as e:
    # Handle cases where 'genai' might have imported but doesn't have 'types'
    logger.warning(f"Failed to access 'google.genai.types' (AttributeError): {e}. Possible incomplete google-genai installation?")

try:
    from google.api_core.exceptions import GoogleAPIError
    _api_error_imported = True
except ImportError as e:
    # This might indicate a problem with google-api-core, a dependency
    logger.warning(f"Failed to import 'google.api_core.exceptions.GoogleAPIError': {e}. Check google-api-core installation.")
    GoogleAPIError = Exception # Keep placeholder

# Set availability flag based on successful imports
GENAI_AVAILABLE = _genai_imported and _types_imported and _api_error_imported

if not GENAI_AVAILABLE:
    logger.warning("One or more Google GenAI components failed to import. Gemini provider will be unavailable.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    DOTENV_AVAILABLE = False

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed. Cost calculation will be skipped. Run `uv add litellm`.")

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# --- Default Model Names ---
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite" # Use cheapest recent vision-enabled Gemini model as default
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022" # Use the cheapest recent vision-enabled claude model as default
DEFAULT_OPENAI_MODEL = "gpt-4.1-nano" # Use the cheapest recent vision-enabled GPT-4 model as default
DEFAULT_OLLAMA_MODEL = "gemma3" # Use a recent vision-enabled model for Ollama as default -- 4b model using base name
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"


# --- Helper Functions ---
def _load_image_bytes(image_input: np.ndarray | str) -> bytes:
    """Loads image from path or converts NumPy array to PNG bytes."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found at path: {image_input}")
        with open(image_input, "rb") as img_file:
            return img_file.read()
    elif isinstance(image_input, np.ndarray):
        is_success, buffer = cv2.imencode(".png", image_input)
        if not is_success:
            raise RuntimeError("Failed to encode NumPy image to PNG format.")
        return buffer.tobytes()
    else:
        raise TypeError("Unsupported image type. Provide path (str) or NumPy array.")

def _get_mime_type(image_path_or_bytes: str | bytes) -> str:
    """Determines the mime type of an image."""
    if isinstance(image_path_or_bytes, str):
        mime_type, _ = mimetypes.guess_type(image_path_or_bytes)
        return mime_type or "application/octet-stream" # Default if cannot guess
    else:
        # Basic check for common types based on magic bytes
        if image_path_or_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        elif image_path_or_bytes.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif image_path_or_bytes.startswith(b'GIF8'):
            return "image/gif"
        elif image_path_or_bytes.startswith(b'RIFF') and image_path_or_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            return "application/octet-stream" # Default fallback

def _encode_image_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to Base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


# --- LLMExtractor Class ---
class LLMExtractor(BaseExtractor):
    """Extractor using LLM APIs: Google Gemini, Ollama, Anthropic, OpenAI."""

    provider: str
    model_name: str
    api_key_env_var: str | None = None
    api_key: str | None = None
    ollama_endpoint: str | None = None
    client: Any | None = None # API client (e.g., genai model, anthropic client, openai client)
    prompt_template: str = "Extract all text verbatim from this image document." # Default prompt

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the extractor based on the provider."""
        super().__init__(config)
        self.provider = self.config.get("provider", "gemini").lower()
        self.model_name = self.config.get("model") # Model must be specified per provider now
        self.prompt_template = self.config.get("prompt", self.prompt_template)

        # Load .env file if available
        if DOTENV_AVAILABLE and load_dotenv:
            loaded_env = load_dotenv()
            logger.debug(f".env file loaded: {loaded_env}")

        if self.provider == "gemini":
            if not GENAI_AVAILABLE:
                raise ImportError("Google GenAI SDK not installed. Run `uv add google-genai`.")
            self.api_key_env_var = self.config.get("api_key_env", "GOOGLE_API_KEY")
            self.api_key = os.getenv(self.api_key_env_var)
            if not self.api_key:
                # Log error before raising
                logger.error(f"Gemini API key env var {self.api_key_env_var} not found or empty.")
                raise ValueError(f"Gemini API key not found in env var {self.api_key_env_var}.")
            else:
                # Optional: Log prefix for confirmation (remove in production if sensitive)
                logger.debug(f"Gemini API key loaded (starts with: {self.api_key[:4]}...).")

            # Note: google-genai uses genai.Client() which auto-configures from env var
            self.model_name = self.model_name or DEFAULT_GEMINI_MODEL
            try:
                # Initialize the main client, relying on GOOGLE_API_KEY env var
                self.client = genai.Client()
                # Optionally, verify model existence here if needed, but typically done during call
                # self.client.get_model(self.model_name) # Example check
                logger.info(f"Gemini client initialized successfully. Will use model: {self.model_name}")
            except Exception as e:
                # Log the specific exception
                logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True) # Add exc_info=True
                raise RuntimeError("Failed to initialize Gemini client") from e

        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic SDK not installed. Run `uv add anthropic`.")
            self.api_key_env_var = self.config.get("api_key_env", "ANTHROPIC_API_KEY")
            self.api_key = os.getenv(self.api_key_env_var)
            if not self.api_key:
                raise ValueError(f"Anthropic API key not found in env var {self.api_key_env_var}.")
            self.model_name = self.model_name or DEFAULT_ANTHROPIC_MODEL
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic client initialized for model: {self.model_name}")
            except Exception as e:
                 logger.error(f"Failed to initialize Anthropic client: {e}")
                 raise RuntimeError("Failed to initialize Anthropic client") from e

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI SDK not installed. Run `uv add openai`.")
            self.api_key_env_var = self.config.get("api_key_env", "OPENAI_API_KEY")
            self.api_key = os.getenv(self.api_key_env_var)
            if not self.api_key:
                raise ValueError(f"OpenAI API key not found in env var {self.api_key_env_var}.")
            self.model_name = self.model_name or DEFAULT_OPENAI_MODEL
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized for model: {self.model_name}")
            except Exception as e:
                 logger.error(f"Failed to initialize OpenAI client: {e}")
                 raise RuntimeError("Failed to initialize OpenAI client") from e

        elif self.provider == "ollama":
            if not REQUESTS_AVAILABLE:
                 raise ImportError("Requests library not installed. Run `uv add requests`.")
            self.model_name = self.model_name or DEFAULT_OLLAMA_MODEL
            self.ollama_endpoint = self.config.get("endpoint", DEFAULT_OLLAMA_ENDPOINT)
            logger.info(f"Ollama configured: Endpoint={self.ollama_endpoint}, Model={self.model_name}")
            # No persistent client needed for requests-based approach

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Choose 'gemini', 'ollama', 'anthropic', or 'openai'.")

        if not self.model_name:
             raise ValueError(f"Model name must be specified in config for provider '{self.provider}'.")


    def extract(self, image: np.ndarray | str) -> ExtractionResult:
        """Extracts text using the configured LLM provider."""
        if self.provider == "gemini":
            return self._extract_gemini(image)
        elif self.provider == "anthropic":
             return self._extract_anthropic(image)
        elif self.provider == "openai":
             return self._extract_openai(image)
        elif self.provider == "ollama":
            return self._extract_ollama(image)
        else:
            raise RuntimeError(f"Invalid provider configured: {self.provider}") # Should not happen

    # --- Provider-Specific Extraction Methods ---

    def _extract_gemini(self, image_input: np.ndarray | str) -> ExtractionResult:
        """Handles extraction using the Google Gemini API (google-genai SDK)."""
        if self.client is None: raise RuntimeError("Gemini client not initialized.")
        if types is None: raise RuntimeError("google.genai.types not available.")

        try:
            image_bytes = _load_image_bytes(image_input)
            mime_type = _get_mime_type(image_bytes) # Get mime type from bytes

            logger.debug(f"Sending image ({mime_type}) to Gemini model {self.model_name}...")

            # Create content part using google.genai.types
            # image_part = types.Part.from_data(data=image_bytes, mime_type=mime_type) # Incorrect method
            # --- Manually create Blob and Part --- #
            image_blob = types.Blob(mime_type=mime_type, data=image_bytes)
            image_part = types.Part(inline_data=image_blob)

            # Define contents for the API call
            contents = [self.prompt_template, image_part]

            # IMPROVEMENT: Add logging before the API call
            logger.debug(f"Attempting Gemini API call. Model: {self.model_name}, Prompt: '{self.prompt_template[:50]}...', Image mime: {mime_type}, Num_parts: {len(contents)}")

            # Generate content using client.models
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
                # Add generation_config here if needed (e.g., temperature)
                # generation_config=types.GenerationConfig(...)
            )

            # IMPROVEMENT: Log the raw response structure (or parts of it) for debugging
            # Use repr() for potentially more detailed structure info, careful with large data
            logger.debug(f"Raw Gemini response received (type: {type(response)}). Full repr(): {repr(response)}")
            # Alternatively log specific attributes if known
            # logger.debug(f"Gemini response parts: {getattr(response, 'parts', 'N/A')}")
            # logger.debug(f"Gemini response text exists: {hasattr(response, 'text')}")
            # logger.debug(f"Gemini response usage metadata: {getattr(response, 'usage_metadata', 'N/A')}")

            extracted_text = ""
            cost = 0.0
            misc_data = {}

            # Extract text (google-genai SDK response structure)
            try:
                # Preferred way according to docs seems to be response.text
                extracted_text = response.text.strip()
            except AttributeError:
                 # Fallback in case .text isn't present (e.g., blocked response)
                 try:
                     if response.parts:
                         extracted_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
                     else:
                         logger.warning(f"Could not extract text from Gemini response. Parts: {response.parts}")
                         extracted_text = "" # Or handle error as needed
                 except Exception as e_parse:
                     logger.error(f"Failed to parse Gemini response parts: {e_parse}. Response: {response}")
                     extracted_text = "" # Ensure it's a string

            # --- FIX: Populate misc_data before cost calculation ---
            if hasattr(response, 'usage_metadata'):
                misc_data['usage_metadata'] = response.usage_metadata
            else:
                logger.warning("Gemini response object does not have 'usage_metadata' attribute.")
            # -------------------------------------------------------

            # Cost calculation using litellm
            cost = 0.0
            if LITELLM_AVAILABLE and misc_data.get('usage_metadata'):
                try:
                    # Extract token counts from usage metadata
                    usage_info = misc_data['usage_metadata']
                    prompt_tokens = getattr(usage_info, 'prompt_token_count', 0)
                    completion_tokens = getattr(usage_info, 'candidates_token_count', 0)

                    # Calculate cost using cost_per_token
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
                    cost = prompt_cost + completion_cost

                    logger.debug(f"Calculated Gemini cost via litellm (cost_per_token): ${cost:.6f} (Prompt: {prompt_tokens}, Completion: {completion_tokens})")
                except Exception as e_cost:
                    logger.warning(f"litellm cost calculation failed for Gemini model {self.model_name}: {e_cost}")
            elif not LITELLM_AVAILABLE:
                logger.debug("litellm not available, skipping Gemini cost calculation.")
            else:
                logger.warning("Gemini response did not contain usage metadata for cost calculation.")
                # cost remains 0.0

            logger.debug(f"Gemini model extracted text (first 50 chars): '{extracted_text[:50]}...'" if extracted_text else "Gemini model extracted empty text.")
            return ExtractionResult(text=extracted_text, latency_seconds=-1, cost=cost, misc=misc_data)

        except GoogleAPIError as e:
            # IMPROVEMENT: Log the detailed API error
            logger.error(f"Gemini API error occurred: Status={getattr(e, 'code', 'N/A')}, Message='{getattr(e, 'message', str(e))}'", exc_info=True)
            # Consider adding more details if available in 'e', e.g., e.details
            raise RuntimeError(f"Gemini API request failed for model {self.model_name}") from e
        except Exception as e:
            # IMPROVEMENT: Log the general exception with traceback
            logger.error(f"Unexpected error during Gemini extraction: {e}", exc_info=True) # Use logger.error and exc_info
            raise RuntimeError(f"Gemini extraction failed for model {self.model_name}") from e

    def _extract_anthropic(self, image_input: np.ndarray | str) -> ExtractionResult:
        """Handles extraction using the Anthropic Claude API."""
        if self.client is None: raise RuntimeError("Anthropic client not initialized.")

        try:
            image_bytes = _load_image_bytes(image_input)
            image_base64 = _encode_image_base64(image_bytes)
            image_media_type = _get_mime_type(image_bytes)

            logger.debug(f"Sending image ({image_media_type}) to Anthropic model {self.model_name}...")

            # Construct message payload for Claude vision models
            # Ref: https://docs.anthropic.com/claude/reference/messages_post
            message_list = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.prompt_template
                        }
                    ],
                }
            ]

            # Use client.messages.create
            # Increase max_tokens if expecting long output, add other params as needed
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096, # Adjust as needed
                messages=message_list,
                # temperature=0.0 # Set temperature if desired
            )

            # --- Log raw Anthropic response --- #
            logger.debug(f"Raw Anthropic response received (type: {type(response)}). Full response object: {response}")

            extracted_text = ""
            cost = 0.0
            misc_data = {}

            # Extract text from response (Claude typically puts it in response.content[0].text)
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                 extracted_text = response.content[0].text.strip()
            else:
                 logger.error(f"Could not extract text from Anthropic response content: {response.content}")

            # --- FIX: Populate misc_data before cost calculation ---
            if hasattr(response, 'usage'):
                misc_data['usage'] = response.usage
            else:
                logger.warning("Anthropic response object does not have 'usage' attribute.")
            # -------------------------------------------------------

            # Cost calculation using litellm
            cost = 0.0
            if LITELLM_AVAILABLE and misc_data.get('usage'):
                try:
                    # Extract token counts from usage object
                    usage_info = misc_data['usage']
                    prompt_tokens = getattr(usage_info, 'input_tokens', 0)
                    completion_tokens = getattr(usage_info, 'output_tokens', 0)

                    # Calculate cost using cost_per_token
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
                    cost = prompt_cost + completion_cost

                    logger.debug(f"Calculated Anthropic cost via litellm (cost_per_token): ${cost:.6f} (Input: {prompt_tokens}, Output: {completion_tokens})")
                except Exception as e_cost:
                    logger.warning(f"litellm cost calculation failed for Anthropic model {self.model_name}: {e_cost}")
            elif not LITELLM_AVAILABLE:
                 logger.debug("litellm not available, skipping Anthropic cost calculation.")
            else:
                 logger.warning("Anthropic response did not contain usage data for cost calculation.")
                 # cost remains 0.0

            logger.debug(f"Anthropic model extracted text (first 50 chars): '{extracted_text[:50]}...'")
            return ExtractionResult(text=extracted_text, latency_seconds=-1, cost=cost, misc=misc_data)

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API request failed for model {self.model_name}") from e
        except Exception as e:
            logger.exception(f"Error during Anthropic extraction: {e}")
            raise RuntimeError(f"Anthropic extraction failed for model {self.model_name}") from e


    def _extract_openai(self, image_input: np.ndarray | str) -> ExtractionResult:
        """Handles extraction using the OpenAI GPT API."""
        if self.client is None: raise RuntimeError("OpenAI client not initialized.")

        try:
            image_bytes = _load_image_bytes(image_input)
            image_base64 = _encode_image_base64(image_bytes)
            # OpenAI generally prefers data URI format for base64 images
            mime_type = _get_mime_type(image_bytes)
            image_data_uri = f"data:{mime_type};base64,{image_base64}"

            logger.debug(f"Sending image ({mime_type}) to OpenAI model {self.model_name}...")

            # Construct message payload for GPT-4 vision models
            # Ref: https://platform.openai.com/docs/guides/vision
            message_list = [
                 {
                     "role": "user",
                     "content": [
                         {"type": "text", "text": self.prompt_template},
                         {
                             "type": "image_url",
                             "image_url": {"url": image_data_uri},
                         },
                     ],
                 }
            ]

            # Use client.chat.completions.create
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                max_tokens=4096, # Adjust as needed
            )

            # --- Log raw OpenAI response --- #
            logger.debug(f"Raw OpenAI response received (type: {type(response)}). Full response object: {response}")

            extracted_text = ""
            cost = 0.0
            misc_data = {}

            # Extract text from response
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 extracted_text = response.choices[0].message.content.strip()
            else:
                 logger.error(f"Could not extract text from OpenAI response choices: {response.choices}")

            # --- FIX: Populate misc_data before cost calculation ---
            if hasattr(response, 'usage'):
                misc_data['usage'] = response.usage
            else:
                logger.warning("OpenAI response object does not have 'usage' attribute.")
            # -------------------------------------------------------

            # Cost calculation using litellm
            cost = 0.0
            if LITELLM_AVAILABLE and misc_data.get('usage'):
                try:
                    # Extract token counts from usage object
                    usage_info = misc_data['usage']
                    prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage_info, 'completion_tokens', 0)

                    # Calculate cost using cost_per_token
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
                    cost = prompt_cost + completion_cost

                    logger.debug(f"Calculated OpenAI cost via litellm (cost_per_token): ${cost:.6f} (Prompt: {prompt_tokens}, Completion: {completion_tokens})")
                except Exception as e_cost:
                    logger.warning(f"litellm cost calculation failed for OpenAI model {self.model_name}: {e_cost}")
            elif not LITELLM_AVAILABLE:
                 logger.debug("litellm not available, skipping OpenAI cost calculation.")
            else:
                 logger.warning("OpenAI response did not contain usage data for cost calculation.")
                 # cost remains 0.0

            logger.debug(f"OpenAI model extracted text (first 50 chars): '{extracted_text[:50]}...'")
            return ExtractionResult(text=extracted_text, latency_seconds=-1, cost=cost, misc=misc_data)

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API request failed for model {self.model_name}") from e
        except Exception as e:
            logger.exception(f"Error during OpenAI extraction: {e}")
            raise RuntimeError(f"OpenAI extraction failed for model {self.model_name}") from e


    def _extract_ollama(self, image_input: np.ndarray | str) -> ExtractionResult:
        """Handles extraction using a local Ollama API endpoint."""
        if not REQUESTS_AVAILABLE: raise ImportError("Requests library not installed.")

        logger.info(f"Connecting to Ollama endpoint: {self.ollama_endpoint}")

        try:
            image_bytes = _load_image_bytes(image_input)
            encoded_image = _encode_image_base64(image_bytes)

            payload = {
                "model": self.model_name,
                "prompt": self.prompt_template,
                "images": [encoded_image],
                "stream": False
            }

            api_url = f"{self.ollama_endpoint.rstrip('/')}/api/generate"
            timeout = self.config.get("timeout", 120)
            logger.debug(f"Sending request to Ollama: {api_url} with model {self.model_name}")

            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()

            response_data = response.json()
            # --- Log raw Ollama response --- #
            logger.debug(f"Raw Ollama response JSON received: {response_data}")
            # -------------------------------- #
            extracted_text = response_data.get("response", "").strip()
            misc_data = {k: v for k, v in response_data.items() if k != 'response'}

            # Ollama cost is typically zero, but we can try litellm if usage is reported
            # Note: Ollama API response structure might vary. misc_data holds the JSON.
            # We need to check if it contains standard token counts.
            cost = 0.0
            ollama_prompt_tokens = misc_data.get('prompt_eval_count', 0)
            ollama_completion_tokens = misc_data.get('eval_count', 0)

            if LITELLM_AVAILABLE and (ollama_prompt_tokens > 0 or ollama_completion_tokens > 0):
                try:
                    cost = litellm.completion_cost(
                        model=self.model_name, # Pass model name
                        prompt_tokens=ollama_prompt_tokens,
                        completion_tokens=ollama_completion_tokens,
                    )
                    # Override cost to 0 for local models, but log the potential cloud cost
                    logger.debug(f"litellm estimated potential cost for Ollama model {self.model_name}: ${cost:.6f} (Assuming cloud pricing. Actual cost is likely $0.00)")
                    cost = 0.0 # Override cost to 0 for local Ollama
                except Exception as e_cost:
                    logger.warning(f"litellm cost calculation failed for Ollama model {self.model_name}: {e_cost}. Assuming $0.00 cost.")
                    cost = 0.0
            else:
                logger.debug(f"Skipping litellm cost calculation for Ollama (litellm unavailable or no token info). Assuming $0.00 cost.")
                cost = 0.0

            logger.debug(f"Ollama model extracted text (first 50 chars): '{extracted_text[:50]}...'")
            return ExtractionResult(text=extracted_text, latency_seconds=-1, cost=cost, misc=misc_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error: {e}")
            raise RuntimeError(f"Ollama API request failed to {api_url}") from e
        except Exception as e:
            logger.exception(f"Error during Ollama extraction: {e}")
            raise RuntimeError(f"Ollama extraction failed for model {self.model_name}") from e 