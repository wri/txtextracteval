#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the EasyOCRExtractor."""

import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np

# Import the class to test
from txtextracteval.extractors.easyocr_engine import EasyOCRExtractor, EASYOCR_AVAILABLE
from txtextracteval.extractors.base import ExtractionResult

@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader') # Patch where it's used
def test_easyocr_extractor_init(MockEasyOCRReaderInstance):
    """Test successful initialization of EasyOCRExtractor."""
    mock_reader = MagicMock() # This is the instance returned by easyocr.Reader()
    MockEasyOCRReaderInstance.return_value = mock_reader

    config = {'languages': ['en', 'fr'], 'gpu': False, 'custom_param': 'value'}
    extractor = EasyOCRExtractor(config)
    
    MockEasyOCRReaderInstance.assert_called_once_with(['en', 'fr'], gpu=False, custom_param='value')
    assert extractor.languages == ['en', 'fr']
    assert extractor.gpu is False
    assert extractor.reader is mock_reader # Check it stores the instance

@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader')
def test_easyocr_extractor_init_defaults(MockEasyOCRReaderInstance):
    """Test initialization with default parameters."""
    mock_reader = MagicMock()
    MockEasyOCRReaderInstance.return_value = mock_reader

    extractor = EasyOCRExtractor({})
    MockEasyOCRReaderInstance.assert_called_once_with(['en'], gpu=True)
    assert extractor.languages == ['en']
    assert extractor.gpu is True
    assert extractor.reader is mock_reader

@patch.dict('sys.modules', {'easyocr': MagicMock()})
def test_easyocr_unavailable_init_raises_import_error():
    """Test that ImportError is raised if EasyOCR is not available during init."""
    # Temporarily set EASYOCR_AVAILABLE to False for this test, even if it is installed
    with patch('txtextracteval.extractors.easyocr_engine.EASYOCR_AVAILABLE', False):
        with pytest.raises(ImportError, match="EasyOCR library not found"): 
            EasyOCRExtractor({})

@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader')
def test_easyocr_extractor_extract_success(MockEasyOCRReaderInstance):
    """Test successful text extraction."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.readtext.return_value = ["Hello world", "This is EasyOCR"]
    MockEasyOCRReaderInstance.return_value = mock_reader_instance

    extractor = EasyOCRExtractor({}) # Initializes with the mocked reader instance
    # extractor.reader is now mock_reader_instance

    image_path = "dummy/path/to/image.png"
    result = extractor.extract(image_path)

    assert isinstance(result, ExtractionResult)
    assert result.text == "Hello world\nThis is EasyOCR"
    assert result.cost == 0.0
    assert result.latency_seconds > 0
    mock_reader_instance.readtext.assert_called_once_with(image_path, detail=0, paragraph=True)

@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader')
def test_easyocr_extractor_extract_numpy_input(MockEasyOCRReaderInstance):
    """Test successful text extraction with a NumPy array as input."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.readtext.return_value = ["NumPy test"]
    MockEasyOCRReaderInstance.return_value = mock_reader_instance
    
    extractor = EasyOCRExtractor({})
    dummy_image_np = np.zeros((10,10), dtype=np.uint8)
    result = extractor.extract(dummy_image_np)

    assert result.text == "NumPy test"
    mock_reader_instance.readtext.assert_called_once_with(dummy_image_np, detail=0, paragraph=True)

@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader')
def test_easyocr_extractor_extract_failure(MockEasyOCRReaderInstance):
    """Test text extraction failure."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.readtext.side_effect = Exception("OCR failed miserably")
    MockEasyOCRReaderInstance.return_value = mock_reader_instance

    extractor = EasyOCRExtractor({})
    image_path = "dummy/path/to/image.png"
    with pytest.raises(RuntimeError, match="EasyOCR extraction failed: OCR failed miserably"):
        extractor.extract(image_path)
    
    # Ensure latency is still recorded even on failure (as it's in finally block)
    # This part is harder to test directly without more complex mocking or checking logs
    # but the structure of extract() method implies it.

@patch('txtextracteval.extractors.easyocr_engine.EASYOCR_AVAILABLE', False)
@patch.dict('sys.modules', {'easyocr': MagicMock()}) # Ensure easyocr is mocked for import
def test_easyocr_import_error_if_not_available():
    """If EASYOCR_AVAILABLE is False, init should raise ImportError."""
    # This test specifically ensures that if the import fails at the module level,
    # the class instantiation will fail as expected.
    with pytest.raises(ImportError, match="EasyOCR library not found"):
        EasyOCRExtractor({})

# It might be useful to have a test for when easyocr.Reader itself fails to initialize
@pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")
@patch('txtextracteval.extractors.easyocr_engine.easyocr.Reader') # Patch where it's used
def test_easyocr_reader_init_failure(MockEasyOCRReaderInstance):
    """Test when easyocr.Reader() initialization fails."""
    MockEasyOCRReaderInstance.side_effect = Exception("Failed to load model")
    with pytest.raises(RuntimeError, match="EasyOCR Reader initialization failed: Failed to load model"):
        EasyOCRExtractor({}) 