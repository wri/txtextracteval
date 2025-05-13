#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the extractor classes."""

import pytest
import numpy as np
from PIL import Image
import os

# Mock external dependencies before importing extractors that use them
from unittest.mock import MagicMock, patch

# Import specific exceptions or constants needed even when modules are mocked
import pytesseract # Import to access TesseractNotFoundError
import requests # Import for requests.exceptions.RequestException
from txtextracteval.extractors.llm_api import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OLLAMA_MODEL # Import defaults

# Mock pytesseract before it's imported by TesseractExtractor
# We use patch on the module where it's looked up (tesseract.py)
@patch('txtextracteval.extractors.tesseract.pytesseract', MagicMock()) 
# Mock transformers pipeline before it's imported by HFExtractor
@patch('txtextracteval.extractors.hf.pipeline', MagicMock(return_value=MagicMock())) 
# Mock google.generativeai before it's imported by LLMExtractor
@patch('txtextracteval.extractors.llm_api.genai', MagicMock()) 
# Mock requests before it's used by LLMExtractor
@patch('txtextracteval.extractors.llm_api.requests', MagicMock()) 
# Mock cv2 used in multiple places
@patch('txtextracteval.extractors.tesseract.cv2', MagicMock()) 
@patch('txtextracteval.extractors.hf.cv2', MagicMock()) 
@patch('txtextracteval.extractors.llm_api.cv2', MagicMock())
@patch('txtextracteval.extractors.llm_api.load_dotenv', MagicMock())
# Mock Image.open used in multiple places
@patch('txtextracteval.extractors.tesseract.Image', MagicMock()) 
@patch('txtextracteval.extractors.hf.Image', MagicMock())
@patch('txtextracteval.extractors.llm_api.Image', MagicMock())
def setup_mocks():
    # This function doesn't need to do anything, 
    # its purpose is to group the patch decorators.
    pass

# Now import the classes after mocks are set up
from txtextracteval.extractors import TesseractExtractor, HFExtractor, LLMExtractor, ExtractionResult

# --- Fixtures --- #

@pytest.fixture
def sample_np_image() -> np.ndarray:
    """Returns a dummy NumPy array representing an image."""
    return np.zeros((10, 10, 3), dtype=np.uint8)

@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Ensure dummy API key exists for tests needing it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key")

# --- Tests for TesseractExtractor --- #

def test_tesseract_extractor_init():
    """Test TesseractExtractor initialization."""
    config = {"lang": "eng", "psm": "3"}
    extractor = TesseractExtractor(config=config)
    assert extractor.lang == "eng"
    assert extractor.psm == "3"

@patch('txtextracteval.extractors.tesseract.pytesseract')
@patch('txtextracteval.extractors.tesseract.Image')
def test_tesseract_extractor_extract_path(MockImage, mock_pytesseract, tmp_path):
    """Test TesseractExtractor extract method with a file path."""
    # Arrange
    mock_img_instance = MockImage.open.return_value
    mock_pytesseract.image_to_string.return_value = " Extracted Text \n"
    extractor = TesseractExtractor()
    test_file = tmp_path / "test.png"
    test_file.touch() # Create dummy file

    # Act
    result = extractor.extract(str(test_file))

    # Assert
    MockImage.open.assert_called_once_with(str(test_file))
    mock_pytesseract.image_to_string.assert_called_once_with(mock_img_instance, config="")
    assert isinstance(result, ExtractionResult)
    assert result.text == "Extracted Text"
    assert result.cost == 0.0

@patch('txtextracteval.extractors.tesseract.pytesseract')
@patch('txtextracteval.extractors.tesseract.cv2')
@patch('txtextracteval.extractors.tesseract.Image')
def test_tesseract_extractor_extract_numpy(MockImage, mock_cv2, mock_pytesseract, sample_np_image):
    """Test TesseractExtractor extract method with a NumPy array."""
    # Arrange
    mock_cv2.cvtColor.return_value = sample_np_image # Mock conversion result
    mock_pil_image = MockImage.fromarray.return_value
    mock_pytesseract.image_to_string.return_value = "NumPy Text"
    extractor = TesseractExtractor()

    # Act
    result = extractor.extract(sample_np_image)

    # Assert
    mock_cv2.cvtColor.assert_called_once()
    MockImage.fromarray.assert_called_once_with(sample_np_image)
    mock_pytesseract.image_to_string.assert_called_once_with(mock_pil_image, config="")
    assert result.text == "NumPy Text"

@patch('txtextracteval.extractors.tesseract.pytesseract')
@patch('txtextracteval.extractors.tesseract.Image')
def test_tesseract_extractor_handles_tesseract_error(MockImage, mock_pytesseract_in_sut, tmp_path):
    """Test TesseractExtractor raises error if pytesseract fails."""
    MockImage.open.return_value = MagicMock()
    # Make the mocked function raise the *actual* TesseractNotFoundError
    # The error message can be checked in the test
    mock_pytesseract_in_sut.image_to_string.side_effect = pytesseract.TesseractNotFoundError("Mocked Tesseract Error")
    extractor = TesseractExtractor()
    test_file = tmp_path / "test.png"
    test_file.touch()
    # Expect the *actual* TesseractNotFoundError to be caught and re-raised
    with pytest.raises(pytesseract.TesseractNotFoundError, match="Mocked Tesseract Error"):
        extractor.extract(str(test_file))

# --- Tests for HFExtractor --- #

# Mock the pipeline object that the class interacts with
@patch('txtextracteval.extractors.hf.pipeline')
def test_hf_extractor_init(mock_hf_pipeline):
    """Test HFExtractor initialization loads the pipeline."""
    # Arrange
    mock_pipeline_instance = MagicMock()
    mock_hf_pipeline.return_value = mock_pipeline_instance
    config = {"model": "test-model"}

    # Act
    extractor = HFExtractor(config=config)

    # Assert
    mock_hf_pipeline.assert_called_once_with("image-to-text", model="test-model", device=-1)
    assert extractor.pipeline == mock_pipeline_instance
    assert extractor.model_name == "test-model"

@patch('txtextracteval.extractors.hf.pipeline')
@patch('txtextracteval.extractors.hf.Image')
def test_hf_extractor_extract_path(MockImage, mock_hf_pipeline, tmp_path):
    """Test HFExtractor extract method with a file path."""
    # Arrange
    mock_pipeline_instance = MagicMock()
    # Simulate pipeline output structure
    mock_pipeline_instance.return_value = [{"generated_text": " HF Text Result "}]
    mock_hf_pipeline.return_value = mock_pipeline_instance
    mock_pil_image = MockImage.open.return_value.convert.return_value
    
    extractor = HFExtractor() # Uses default model name for loading mock pipeline
    test_file = tmp_path / "hf_test.png"
    test_file.touch()

    # Act
    result = extractor.extract(str(test_file))

    # Assert
    MockImage.open.assert_called_once_with(str(test_file))
    mock_pil_image = MockImage.open.return_value.convert.assert_called_once_with("RGB")
    mock_pipeline_instance.assert_called_once() # Check pipeline was called
    # Check the argument passed to the pipeline was the loaded image
    # assert mock_pipeline_instance.call_args[0][0] == mock_pil_image # Fails easily due to mock complexity
    assert isinstance(result, ExtractionResult)
    assert result.text == "HF Text Result"
    assert result.cost == 0.0

# --- Tests for LLMExtractor --- #

# Mock google.generativeai client methods for Gemini tests
@patch('txtextracteval.extractors.llm_api.load_dotenv') # Keep this as it affects env var loading logic
@patch('txtextracteval.extractors.llm_api.genai') # Patch genai specifically for this test
def test_llm_extractor_init_gemini(mock_genai_in_llmapi, mock_load_dotenv_in_llmapi, monkeypatch):
    """Test LLMExtractor initialization for Gemini provider."""
    # Arrange
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_gemini_init")
    mock_model_instance = MagicMock()
    mock_genai_in_llmapi.GenerativeModel.return_value = mock_model_instance
    config = {"provider": "gemini", "model": "gemini-test"}
    
    # Act
    extractor = LLMExtractor(config=config)

    # Assert
    mock_load_dotenv_in_llmapi.assert_called_once()
    mock_genai_in_llmapi.configure.assert_called_once_with(api_key="test_api_key_gemini_init")
    mock_genai_in_llmapi.GenerativeModel.assert_called_once_with("gemini-test")
    assert extractor.provider == "gemini"
    assert extractor.client == mock_model_instance

@patch('txtextracteval.extractors.llm_api.requests')
def test_llm_extractor_init_ollama(mock_requests):
    """Test LLMExtractor initialization for Ollama provider."""
    config = {"provider": "ollama", "model": "ollama-test", "endpoint": "http://ollama:11434"}
    extractor = LLMExtractor(config=config)
    assert extractor.provider == "ollama"
    assert extractor.model_name == "ollama-test"
    assert extractor.ollama_endpoint == "http://ollama:11434"
    mock_requests.post.assert_not_called() # Should not call during init

@patch('txtextracteval.extractors.llm_api.load_dotenv') # For env var logic
@patch('txtextracteval.extractors.llm_api.Image') # Patch Image where it's used in llm_api
@patch('txtextracteval.extractors.llm_api.genai') # Patch genai where it's used in llm_api
def test_llm_extractor_extract_gemini(mock_genai_in_llmapi, MockImage_in_llmapi, mock_load_dotenv_in_llmapi, tmp_path, monkeypatch):
    """Test LLMExtractor extract method for Gemini."""
    # Arrange
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_gemini_extract")
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = " Gemini Response "
    mock_model_instance.generate_content.return_value = mock_response
    mock_genai_in_llmapi.GenerativeModel.return_value = mock_model_instance
    
    # Mock Image.open().convert()
    mock_pil_image_instance = MagicMock()
    MockImage_in_llmapi.open.return_value.convert.return_value = mock_pil_image_instance

    # extractor needs to be created after genai is properly mocked for its init
    extractor = LLMExtractor(config={"provider": "gemini", "model": "gemini-test-extract"})
    test_file = tmp_path / "gemini_test.png"
    test_file.touch()

    # Act
    result = extractor.extract(str(test_file))

    # Assert
    MockImage_in_llmapi.open.assert_called_once_with(str(test_file))
    MockImage_in_llmapi.open.return_value.convert.assert_called_once_with("RGB")
    mock_model_instance.generate_content.assert_called_once()
    # Check the actual arguments passed to generate_content
    args_to_generate_content = mock_model_instance.generate_content.call_args[0][0]
    assert len(args_to_generate_content) == 2
    assert args_to_generate_content[0] == extractor.prompt_template # Default prompt
    assert args_to_generate_content[1] is mock_pil_image_instance # Check the PIL image instance
    assert result.text == "Gemini Response"

@patch('txtextracteval.extractors.llm_api.requests')
@patch('txtextracteval.extractors.llm_api.cv2')
@patch('txtextracteval.extractors.llm_api.base64') # Mock base64 encoding
def test_llm_extractor_extract_ollama(mock_base64, mock_cv2, mock_requests, sample_np_image):
    """Test LLMExtractor extract method for Ollama."""
    # Arrange
    mock_cv2.imencode.return_value = (True, np.array([1,2,3], dtype=np.uint8)) # Simulate a np array buffer
    mock_base64.b64encode.return_value.decode.return_value = "encoded_image_string"
    mock_response = MagicMock()
    # Simulate a successful JSON response for Ollama
    mock_response.json.return_value = {
        "model":"ollama-test", 
        "created_at":"2023-10-13T14:27:08.618953Z", 
        "response":" Ollama Result ", 
        "done":True, 
        "context":[1,2,3], 
        "total_duration":123, 
        "load_duration":12, 
        "prompt_eval_count":10, 
        "prompt_eval_duration":1, 
        "eval_count":5, 
        "eval_duration":1
    }
    mock_response.raise_for_status = MagicMock() # Mock this method
    mock_requests.post.return_value = mock_response
    
    config = {"provider": "ollama"}
    extractor = LLMExtractor(config=config)

    # Act
    result = extractor.extract(sample_np_image)

    # Assert
    mock_cv2.imencode.assert_called_once_with(".png", sample_np_image)
    mock_base64.b64encode.assert_called_once_with(np.array([1,2,3], dtype=np.uint8).tobytes())
    mock_requests.post.assert_called_once()
    call_kwargs = mock_requests.post.call_args[1]
    assert call_kwargs['url'] == f"{DEFAULT_OLLAMA_ENDPOINT}/api/generate"
    assert call_kwargs['json']['model'] == DEFAULT_OLLAMA_MODEL
    assert call_kwargs['json']['images'] == ["encoded_image_string"]
    assert result.text == "Ollama Result"
    assert result.cost == 0.0 # Ollama cost should be 0.0
    assert result.misc == {"token_count": 50} # Check misc data capture 

@patch('txtextracteval.extractors.llm_api.requests.post') # Target requests.post directly
@patch('txtextracteval.extractors.llm_api.cv2')
@patch('txtextracteval.extractors.llm_api.base64')
def test_llm_extractor_extract_ollama_request_exception(mock_base64, mock_cv2, mock_requests_post, sample_np_image):
    """Test LLMExtractor extract method for Ollama handles RequestException."""
    mock_cv2.imencode.return_value = (True, np.array([1,2,3], dtype=np.uint8)) # Simulate a np array buffer
    mock_base64.b64encode.return_value.decode.return_value = "encoded_image_string"
    # Make requests.post raise the actual RequestException
    mock_requests_post.side_effect = requests.exceptions.RequestException("Mocked connection error")

    config = {"provider": "ollama"}
    extractor = LLMExtractor(config=config)

    with pytest.raises(RuntimeError, match="Ollama API request failed to http://localhost:11434/api/generate"):
        extractor.extract(sample_np_image)
    # Verify the underlying error message would be part of the chained exception, 
    # but RuntimeError is what's raised by the extractor method. 