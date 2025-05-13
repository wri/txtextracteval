#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for image transformation functions."""

import pytest
import numpy as np
import cv2
from txtextracteval.transforms import (
    apply_gaussian_blur, adjust_brightness, rotate_image, reduce_resolution,
    deskew_image, apply_ocr_prep_pipeline, crop_border
)
from unittest.mock import patch, ANY
import logging

# --- Fixtures --- #

@pytest.fixture
def sample_image() -> np.ndarray:
    """Creates a simple 10x10 grayscale image (numpy array)."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 128 # Add a gray square
    return img

@pytest.fixture
def sample_color_image() -> np.ndarray:
    """Creates a simple 10x10 color image (numpy array)."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8, 0] = 255 # Blue channel
    img[3:7, 3:7, 1] = 128 # Green channel
    return img

# --- Tests for apply_gaussian_blur --- #

def test_gaussian_blur_runs(sample_image):
    """Test that Gaussian blur runs without error and returns same shape."""
    blurred = apply_gaussian_blur(sample_image, kernel_size=3)
    assert blurred.shape == sample_image.shape
    assert blurred.dtype == sample_image.dtype
    # Check if image is actually blurred (e.g., pixel values changed)
    assert not np.array_equal(blurred, sample_image) # More robust check than sum

def test_gaussian_blur_invalid_kernel(sample_image):
    """Test blur with invalid kernel size (even, zero, negative)."""
    # Should default to 5x5 and run
    blurred_even = apply_gaussian_blur(sample_image, kernel_size=4)
    assert blurred_even.shape == sample_image.shape
    blurred_zero = apply_gaussian_blur(sample_image, kernel_size=0)
    assert blurred_zero.shape == sample_image.shape
    blurred_neg = apply_gaussian_blur(sample_image, kernel_size=-3)
    assert blurred_neg.shape == sample_image.shape

def test_gaussian_blur_color(sample_color_image):
    """Test Gaussian blur on a color image."""
    blurred = apply_gaussian_blur(sample_color_image, kernel_size=3)
    assert blurred.shape == sample_color_image.shape
    assert blurred.dtype == sample_color_image.dtype
    assert not np.array_equal(blurred, sample_color_image)

# --- Tests for adjust_brightness --- #

def test_adjust_brightness_increase(sample_image):
    """Test increasing brightness."""
    brighter = adjust_brightness(sample_image, factor=1.5)
    assert brighter.shape == sample_image.shape
    assert brighter.dtype == sample_image.dtype
    assert np.mean(brighter) > np.mean(sample_image) # Mean should increase
    assert np.all(brighter <= 255) # Should clip at 255

def test_adjust_brightness_decrease(sample_image):
    """Test decreasing brightness."""
    darker = adjust_brightness(sample_image, factor=0.5)
    assert darker.shape == sample_image.shape
    assert darker.dtype == sample_image.dtype
    assert np.mean(darker) < np.mean(sample_image) # Mean should decrease
    assert np.all(darker >= 0) # Should clip at 0

def test_adjust_brightness_no_change(sample_image):
    """Test brightness with factor 1.0 (no change)."""
    same = adjust_brightness(sample_image, factor=1.0)
    assert np.array_equal(same, sample_image)

def test_adjust_brightness_color(sample_color_image):
    """Test brightness adjustment on a color image."""
    brighter = adjust_brightness(sample_color_image, factor=1.2)
    assert brighter.shape == sample_color_image.shape
    assert np.mean(brighter) > np.mean(sample_color_image)

# --- Tests for rotate_image --- #

def test_rotate_image_runs(sample_image):
    """Test basic rotation runs and changes shape."""
    rotated = rotate_image(sample_image, angle=45)
    assert rotated.shape != sample_image.shape # Shape should change to fit rotation
    assert rotated.dtype == sample_image.dtype

def test_rotate_image_90(sample_image):
    """Test 90-degree rotation (shape should be transposed)."""
    rotated = rotate_image(sample_image, angle=90)
    # For a square image rotated 90 deg, the new bounding box is the same size
    # but the content is rotated. Let's check a corner value if possible.
    # A more robust check would compare against a known rotated result.
    assert rotated.shape == sample_image.shape[::-1] # Transposed shape for non-square
    # For square: assert rotated.shape == sample_image.shape
    assert not np.array_equal(rotated, sample_image)

def test_rotate_image_0(sample_image):
    """Test rotation by 0 degrees (should not change)."""
    rotated = rotate_image(sample_image, angle=0)
    # Rotation by 0 might still cause minor interpolation differences or size change if not handled perfectly.
    # Let's check if shape is the same and content is very close.
    assert rotated.shape == sample_image.shape
    assert np.allclose(rotated, sample_image, atol=1) # Allow minor diff due to warpAffine

def test_rotate_image_color(sample_color_image):
    """Test rotation on a color image."""
    rotated = rotate_image(sample_color_image, angle=-30)
    assert rotated.shape != sample_color_image.shape
    assert rotated.dtype == sample_color_image.dtype
    assert rotated.shape[2] == 3 # Should still have 3 color channels 

# --- Tests for reduce_resolution --- #

def test_reduce_resolution_valid(sample_image):
    """Test resolution reduction with a valid scale factor."""
    original_height, original_width = sample_image.shape[:2]
    scale_factor = 0.5
    resized = reduce_resolution(sample_image, scale_factor=scale_factor)
    assert resized.shape[0] == int(original_height * scale_factor)
    assert resized.shape[1] == int(original_width * scale_factor)
    assert resized.dtype == sample_image.dtype

def test_reduce_resolution_valid_color(sample_color_image):
    """Test resolution reduction on a color image."""
    original_height, original_width, channels = sample_color_image.shape
    scale_factor = 0.7
    resized = reduce_resolution(sample_color_image, scale_factor=scale_factor)
    assert resized.shape[0] == int(original_height * scale_factor)
    assert resized.shape[1] == int(original_width * scale_factor)
    assert resized.shape[2] == channels
    assert resized.dtype == sample_color_image.dtype

def test_reduce_resolution_invalid_scale_factor(sample_image, caplog):
    """Test with invalid scale factors (should default to 0.5)."""
    original_height, original_width = sample_image.shape[:2]
    expected_height = int(original_height * 0.5)
    expected_width = int(original_width * 0.5)

    # Test scale_factor = 0
    resized_zero = reduce_resolution(sample_image, scale_factor=0)
    assert "Invalid scale_factor 0. Must be > 0.0 and <= 1.0. Using 0.5." in caplog.text
    assert resized_zero.shape[0] == expected_height
    assert resized_zero.shape[1] == expected_width
    caplog.clear()

    # Test scale_factor > 1
    resized_large = reduce_resolution(sample_image, scale_factor=1.5)
    assert "Invalid scale_factor 1.5. Must be > 0.0 and <= 1.0. Using 0.5." in caplog.text
    assert resized_large.shape[0] == expected_height
    assert resized_large.shape[1] == expected_width
    caplog.clear()

    # Test negative scale_factor
    resized_neg = reduce_resolution(sample_image, scale_factor=-0.5)
    assert "Invalid scale_factor -0.5. Must be > 0.0 and <= 1.0. Using 0.5." in caplog.text
    assert resized_neg.shape[0] == expected_height
    assert resized_neg.shape[1] == expected_width

def test_reduce_resolution_scale_factor_one(sample_image):
    """Test with scale_factor = 1.0 (dimensions should be the same)."""
    resized = reduce_resolution(sample_image, scale_factor=1.0)
    assert resized.shape == sample_image.shape
    # Content might be slightly different due to interpolation, check with allclose
    assert np.allclose(resized, sample_image, atol=1)

def test_reduce_resolution_empty_image(caplog):
    """Test with an empty image."""
    empty_image = np.array([])
    resized = reduce_resolution(empty_image, scale_factor=0.5)
    assert "Input image is empty or None." in caplog.text
    assert resized.size == 0 # Should return the same empty image

def test_reduce_resolution_none_image(caplog):
    """Test with a None image."""
    resized = reduce_resolution(None, scale_factor=0.5)
    assert "Input image is empty or None." in caplog.text
    assert resized is None # Should return None

def test_reduce_resolution_too_small_output(caplog):
    """Test when scale_factor results in zero dimensions."""
    small_image = np.zeros((1, 1), dtype=np.uint8) # 1x1 image
    # Scale factor that makes new width/height zero
    resized = reduce_resolution(small_image, scale_factor=0.1)
    assert "Calculated new dimensions (0x0) are too small. Skipping reduction." in caplog.text
    assert np.array_equal(resized, small_image) # Should return original

    small_image_two = np.zeros((3, 3), dtype=np.uint8)
    resized_two = reduce_resolution(small_image_two, scale_factor=0.1) # Results in 0x0
    assert "Calculated new dimensions (0x0) are too small. Skipping reduction." in caplog.text
    assert np.array_equal(resized_two, small_image_two)

# --- Tests for deskew_image --- #

def test_deskew_image_empty_or_none(caplog):
    """Test deskew_image with empty or None input."""
    assert deskew_image(np.array([])) is not None # Should return empty array
    assert "Input image is empty or None" in caplog.text
    caplog.clear()
    assert deskew_image(None) is None
    assert "Input image is empty or None" in caplog.text

def test_deskew_image_unsupported_format(sample_image, caplog):
    """Test with unsupported image formats (e.g. >3 channels)."""
    unsupported_img = np.zeros((10,10,1,1), dtype=np.uint8)
    result = deskew_image(unsupported_img)
    assert np.array_equal(result, unsupported_img)
    assert "Unsupported image format for deskewing" in caplog.text

@patch('txtextracteval.transforms.opencv_transforms.cv2.findContours')
@patch('txtextracteval.transforms.opencv_transforms.cv2.minAreaRect')
@patch('txtextracteval.transforms.opencv_transforms.cv2.warpAffine')
def test_deskew_image_no_contours(mock_warp, mock_min_rect, mock_contours, sample_image, caplog):
    """Test deskew_image when no contours are found."""
    caplog.set_level(logging.INFO) # Ensure INFO logs are captured
    mock_contours.return_value = ([], None) # No contours
    result = deskew_image(sample_image)
    assert np.array_equal(result, sample_image) # Should return original
    assert "No suitable contours found or angles detected" in caplog.text
    mock_warp.assert_not_called()

@patch('txtextracteval.transforms.opencv_transforms.cv2.findContours')
@patch('txtextracteval.transforms.opencv_transforms.cv2.minAreaRect')
@patch('txtextracteval.transforms.opencv_transforms.cv2.warpAffine')
def test_deskew_image_angle_too_small(mock_warp, mock_min_rect, mock_contours, sample_image, caplog):
    """Test deskew_image when detected median angle is too small."""
    caplog.set_level(logging.INFO) # Ensure INFO logs are captured
    # Mock contours and minAreaRect to produce a very small angle
    # Contour data: (center_x, center_y), (width, height), angle
    mock_contours.return_value = ([np.array([[[1,1],[1,2],[2,2],[2,1]]], dtype=np.int32)], None) # A single contour
    mock_min_rect.return_value = ((5,5), (2,2), 0.05) # angle = 0.05 degrees
    
    result = deskew_image(sample_image)
    assert np.array_equal(result, sample_image)
    assert "Median angle 0.05 is too small. No rotation applied." in caplog.text
    mock_warp.assert_not_called()

@patch('txtextracteval.transforms.opencv_transforms.cv2.findContours')
@patch('txtextracteval.transforms.opencv_transforms.cv2.minAreaRect')
@patch('txtextracteval.transforms.opencv_transforms.cv2.warpAffine')
def test_deskew_image_applies_rotation(mock_warp, mock_min_rect, mock_contours, sample_image, caplog):
    """Test deskew_image applies rotation when a valid angle is detected."""
    caplog.set_level(logging.INFO) # Ensure INFO and DEBUG logs are captured for this test
    caplog.set_level(logging.DEBUG) # Specifically for the detected median angle debug log

    mock_contours.return_value = ([np.array([[[1,1],[1,5],[5,5],[5,1]]], dtype=np.int32)], None)
    # rect: ((center_x, center_y), (width, height), angle) -> angle in [-90, 0)
    mock_min_rect.return_value = ((50,50), (20,10), -5.0) # Simulate angle of -5.0 degrees
    
    # IMPORTANT: The actual deskew_image function uses the input `image` for warpAffine,
    # not the grayscaled/thresholded one. So mock_warp should return a modification of `sample_image`.
    expected_deskewed_image = sample_image + 10 # Simulate a distinctly different image
    mock_warp.return_value = expected_deskewed_image 

    result = deskew_image(sample_image)
    assert not np.array_equal(result, sample_image) # Check image content has changed
    assert np.array_equal(result, expected_deskewed_image) # Check it's the image from warpAffine
    assert mock_warp.called
    # Check the log message for the angle.
    assert "Successfully rotated image by -5.00 degrees" in caplog.text
    assert "Deskew: Detected median angle of -5.00 degrees" in caplog.text # Check DEBUG log

# --- Tests for apply_ocr_prep_pipeline --- #

@patch('txtextracteval.transforms.opencv_transforms.apply_gaussian_blur')
@patch('txtextracteval.transforms.opencv_transforms.deskew_image')
@patch('txtextracteval.transforms.opencv_transforms.crop_border')
def test_apply_ocr_prep_pipeline_calls_all_steps(mock_crop, mock_deskew, mock_blur, sample_image):
    """Test that apply_ocr_prep_pipeline calls blur, deskew, and crop."""
    # Mock the transform functions to return the image unchanged but allow checking calls
    mock_blur.return_value = sample_image
    mock_deskew.return_value = sample_image
    mock_crop.return_value = sample_image

    _ = apply_ocr_prep_pipeline(sample_image, denoise_ksize=3, crop_frac=0.01)

    mock_blur.assert_called_once_with(ANY, kernel_size=3)
    mock_deskew.assert_called_once_with(ANY) # ANY because blur might have modified it
    mock_crop.assert_called_once_with(ANY, crop_fraction=0.01)

@patch('txtextracteval.transforms.opencv_transforms.deskew_image')
def test_apply_ocr_prep_pipeline_skip_steps(mock_deskew, sample_image):
    """Test that steps are skipped if params are zero."""
    # Mock deskew as it's always called
    mock_deskew.return_value = sample_image
    
    # Test skipping blur and crop
    with patch('txtextracteval.transforms.opencv_transforms.apply_gaussian_blur') as mock_blur, \
         patch('txtextracteval.transforms.opencv_transforms.crop_border') as mock_crop:
        
        _ = apply_ocr_prep_pipeline(sample_image, denoise_ksize=0, crop_frac=0)
        mock_blur.assert_not_called()
        mock_deskew.assert_called_once() # Deskew should still be called
        mock_crop.assert_not_called() 