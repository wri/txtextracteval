#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image transformation functions using OpenCV."""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies Gaussian Blur to an image.

    Args:
        image: Input image as a NumPy array.
        kernel_size: Size of the Gaussian kernel (must be positive and odd).

    Returns:
        Blurred image as a NumPy array.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        logger.warning(f"Gaussian kernel size must be positive and odd, got {kernel_size}. Using default 5.")
        kernel_size = 5
    try:
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        logger.debug(f"Applied Gaussian blur with kernel size {kernel_size}.")
        return blurred_image
    except Exception as e:
        logger.error(f"Error applying Gaussian blur: {e}")
        return image # Return original image on error

def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjusts the brightness of an image.

    Args:
        image: Input image as a NumPy array.
        factor: Brightness adjustment factor. > 1 increases brightness, < 1 decreases.

    Returns:
        Brightness-adjusted image as a NumPy array.
    """
    try:
        # Convert to float32 to avoid data loss during multiplication
        img_float = image.astype(np.float32)
        # Multiply by the factor
        adjusted_image = np.clip(img_float * factor, 0, 255)
        # Convert back to original dtype (e.g., uint8)
        adjusted_image = adjusted_image.astype(image.dtype)
        logger.debug(f"Adjusted brightness by factor {factor}.")
        return adjusted_image
    except Exception as e:
        logger.error(f"Error adjusting brightness: {e}")
        return image # Return original image on error

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image by a specified angle.

    Args:
        image: Input image as a NumPy array.
        angle: Angle of rotation in degrees (positive = counter-clockwise).

    Returns:
        Rotated image as a NumPy array. The image size may change to accommodate the rotation.
    """
    try:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate the new bounding box size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation and return the image
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
        logger.debug(f"Rotated image by {angle} degrees.")
        return rotated_image
    except Exception as e:
        logger.error(f"Error rotating image: {e}")
        return image # Return original image on error

def add_noise(image: np.ndarray, mean: float = 0.0, stddev: float = 25.0) -> np.ndarray:
    """Adds Gaussian noise to an image.

    Args:
        image: Input image as a NumPy array.
        mean: Mean of the Gaussian noise distribution.
        stddev: Standard deviation of the Gaussian noise distribution.

    Returns:
        Image with added Gaussian noise.
    """
    try:
        if image.dtype != np.uint8:
            logger.warning("Image dtype is not uint8, noise addition might be inaccurate. Converting to float.")
            image_float = image.astype(np.float32)
        else:
            image_float = image.astype(np.float32) # Work with float for noise addition

        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = image_float + noise
        noisy_image = np.clip(noisy_image, 0, 255) # Clip values to valid range
        noisy_image = noisy_image.astype(image.dtype) # Convert back to original type
        logger.debug(f"Added Gaussian noise with mean={mean}, stddev={stddev}.")
        return noisy_image
    except Exception as e:
        logger.error(f"Error adding noise: {e}")
        return image # Return original image on error

def crop_border(image: np.ndarray, crop_fraction: float = 0.05) -> np.ndarray:
    """Crops a fraction of the border from all sides of the image.

    Args:
        image: Input image as a NumPy array.
        crop_fraction: Fraction (0.0 to 0.5) of width/height to crop from each side.
                       For example, 0.05 means crop 5% from top, bottom, left, right.

    Returns:
        Cropped image as a NumPy array.
    """
    if not 0.0 <= crop_fraction < 0.5:
        logger.warning(f"Crop fraction must be between 0.0 and 0.5, got {crop_fraction}. Using 0.05.")
        crop_fraction = 0.05

    try:
        h, w = image.shape[:2]
        crop_h = int(h * crop_fraction)
        crop_w = int(w * crop_fraction)

        if h <= 2 * crop_h or w <= 2 * crop_w:
            logger.warning("Crop fraction too large for image dimensions, skipping crop.")
            return image

        cropped_image = image[crop_h:h - crop_h, crop_w:w - crop_w]
        logger.debug(f"Cropped {crop_fraction*100:.1f}% border from image.")
        return cropped_image
    except Exception as e:
        logger.error(f"Error cropping border: {e}")
        return image # Return original image on error

def apply_skew(image: np.ndarray, skew_angle_x: float = 0.0, skew_angle_y: float = 0.0) -> np.ndarray:
    """Applies a shear transformation (skew) to the image.

    Args:
        image: Input image as a NumPy array.
        skew_angle_x: Horizontal skew angle in degrees. Positive values shear right.
        skew_angle_y: Vertical skew angle in degrees. Positive values shear down.

    Returns:
        Skewed image as a NumPy array. Size might change.
    """
    try:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Convert angles to radians for tan function
        skew_x_rad = np.deg2rad(skew_angle_x)
        skew_y_rad = np.deg2rad(skew_angle_y)

        # Shear matrix
        # M = [[1, tan(skew_x), 0],
        #      [tan(skew_y), 1, 0]]
        M = np.float32([[1, np.tan(skew_x_rad), 0],
                        [np.tan(skew_y_rad), 1, 0]])

        # Calculate new bounding box (similar to rotation)
        # Transform corners
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        transformed_corners = cv2.transform(np.array([corners]), M)[0]

        # Find min/max x and y
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        new_w = int(np.ceil(x_max - x_min))
        new_h = int(np.ceil(y_max - y_min))

        # Adjust the transformation matrix for translation (to keep image centered)
        M[0, 2] -= x_min
        M[1, 2] -= y_min

        # Apply the affine transformation
        skewed_image = cv2.warpAffine(image, M, (new_w, new_h))
        logger.debug(f"Applied skew with angles x={skew_angle_x}, y={skew_angle_y}.")
        return skewed_image
    except Exception as e:
        logger.error(f"Error applying skew: {e}")
        return image

def reduce_resolution(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    """Reduces the resolution of an image by a given scale factor.

    Args:
        image: Input image as a NumPy array.
        scale_factor: Factor by which to scale the image resolution (e.g., 0.5 for 50%).
                      Must be > 0.0 and <= 1.0.

    Returns:
        Resolution-reduced image as a NumPy array.
    """
    if not 0.0 < scale_factor <= 1.0:
        logger.warning(f"Invalid scale_factor {scale_factor}. Must be > 0.0 and <= 1.0. Using 0.5.")
        scale_factor = 0.5

    try:
        if image is None or image.size == 0:
            logger.warning("Input image is empty or None. Skipping resolution reduction.")
            return image

        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        if new_width == 0 or new_height == 0:
            logger.warning(f"Calculated new dimensions ({new_width}x{new_height}) are too small. Skipping reduction.")
            return image

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.debug(f"Reduced image resolution by scale factor {scale_factor} to {new_width}x{new_height}.")
        return resized_image
    except Exception as e:
        logger.error(f"Error reducing resolution: {e}")
        return image # Return original image on error

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Automatically deskews an image by detecting the dominant text orientation.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Deskewed image as a NumPy array, or original image if deskewing fails or is not needed.
    """
    logger.debug("Attempting to deskew image.")
    try:
        if image is None or image.size == 0:
            logger.warning("Input image is empty or None. Skipping deskew.")
            return image

        # Ensure it's a 2D grayscale image for processing
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4: # BGRA
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            logger.warning("Unsupported image format for deskewing. Requires grayscale or BGR/BGRA. Returning original.")
            return image

        # Binarize the image - Otsu's thresholding is good for bimodal images like text
        # Apply a slight Gaussian blur before thresholding to reduce noise and improve Otsu's method
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use the inverted threshold image (text as white pixels) to find contours
        # Dilate to connect text components slightly for better angle detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(thresh, kernel, iterations=2) # iterations can be tuned

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for cnt in contours:
            # Filter small contours that are likely noise
            if cv2.contourArea(cnt) < 100: # This threshold may need tuning
                continue
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            # cv2.minAreaRect returns angles in [-90, 0).
            # If the box is more vertical, angle is closer to -90.
            # If the box is more horizontal, angle is closer to 0.
            # We want to correct small skews, typically for horizontal text.
            if rect[1][0] < rect[1][1]: # width < height, box is predominantly vertical
                angle = angle + 90
            else: # width >= height, box is predominantly horizontal
                # angle is already in a good range for horizontal text skew, e.g. -5 degrees
                pass 
            
            # We are interested in small skew angles, e.g., -45 to 45 degrees
            # If angle is e.g. -85 (nearly vertical text fragment considered horizontal), that's not what we want.
            if abs(angle) <= 45: # Consider only angles within a reasonable skew range
                 angles.append(angle)

        if not angles:
            logger.info("Deskew: No suitable contours found or angles detected. Returning original image.")
            return image

        median_angle = np.median(angles)
        logger.debug(f"Deskew: Detected median angle of {median_angle:.2f} degrees.")

        # If the median angle is very close to 0, no significant skew detected.
        if abs(median_angle) < 0.1: # Threshold for non-skew
            logger.info(f"Deskew: Median angle {median_angle:.2f} is too small. No rotation applied.")
            return image
        
        # Get rotation matrix and rotate the original grayscale image (or color if preferred)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calculate new bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation on the original image (color or grayscale, whatever was input)
        # Using original `image` here, not `gray` if original was color
        deskewed_image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # BORDER_REPLICATE can be changed to BORDER_CONSTANT with a borderValue if needed

        logger.info(f"Deskew: Successfully rotated image by {median_angle:.2f} degrees.")
        return deskewed_image

    except Exception as e:
        logger.error(f"Error during image deskewing: {e}")
        return image # Return original image on error

def apply_ocr_prep_pipeline(image: np.ndarray, denoise_ksize: int = 3, crop_frac: float = 0.02) -> np.ndarray:
    """Applies a standard OCR preprocessing pipeline.

    Current pipeline: Gaussian Denoise -> Auto Deskew -> Crop Border.

    Args:
        image: Input image as a NumPy array.
        denoise_ksize: Kernel size for Gaussian blur denoising (must be positive & odd).
        crop_frac: Fraction to crop from border (0.0 to 0.5).

    Returns:
        Preprocessed image.
    """
    logger.info(f"Applying OCR Prep Pipeline (Denoise k={denoise_ksize}, Crop f={crop_frac}).")
    processed_image = image
    try:
        # 1. Denoise (using Gaussian Blur)
        if denoise_ksize > 0:
            processed_image = apply_gaussian_blur(processed_image, kernel_size=denoise_ksize)

        # 2. Deskew
        logger.debug("Applying deskewing in OCR prep pipeline.")
        processed_image = deskew_image(processed_image)

        # 3. Crop Border
        if crop_frac > 0:
            processed_image = crop_border(processed_image, crop_fraction=crop_frac)

        logger.info("OCR Prep Pipeline finished.")
        return processed_image
    except Exception as e:
        logger.error(f"Error during OCR prep pipeline: {e}")
        return image # Return original image on pipeline error

