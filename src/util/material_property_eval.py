import cv2
import torch
import matplotlib
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog, messagebox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector

crop_coords = None
pixel_size_um = None


def calculate_pixel_size(image, image_size_um=2.0):
    """Calculate the pixel size in micrometers based on the image dimensions."""
    image_width_pixels = image.shape[1]
    pixel_size_um = image_size_um / image_width_pixels
    return pixel_size_um


def process_image(data, height, width, pixel_size_um: float = 2.0) -> dict:
    """
    Process a 2D material with shape: [H, W]

    Returns
    ---
    - Dictionary of material properties
    """

    # Calculate pixel size
    pixel_size_um = calculate_pixel_size(data)

    # Normalize data
    normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Ensure the image is in float format for processing
    img = data.astype(np.float32)
    h, w = data.shape
    # Flattening: Fit a 1st-order polynomial (linear) to each row and subtract
    flattened_img = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        row = img[i, :]
        x = np.arange(w)

        # Fit a linear polynomial (1st order)
        p = np.polyfit(x, row, 1)  # Fit line y = ax + b

        # Compute background
        background = np.polyval(p, x)

        # Subtract the background from the row
        flattened_img[i, :] = row - background

        # flattened_img[i, :] -= np.min(flattened_img[i, :])  # Shift to keep variations

    # Normalize the image for display
    flattened_img = cv2.normalize(flattened_img, None, 0, 255, cv2.NORM_MINMAX)
    flattened_img = np.uint8(flattened_img)

    # Calculate the average current
    average_current = np.mean(data) / 1e-9

    # Define the coverage threshold
    threshold = 120 * 1e-12

    # Create a binary mask for coverage (1 = covered, 0 = uncovered)
    coverage_mask = (data > threshold).astype(int)

    # Calculate coverage percentage
    coverage_percentage = (np.sum(coverage_mask) / coverage_mask.size) * 100

    # Create a color map: Light blue for coverage, light grey for uncovered
    colors = ["black", "mistyrose"]
    cmap = LinearSegmentedColormap.from_list("coverage_map", colors, N=2)

    # Plot the coverage map
    x = np.linspace(0, 2, data.shape[1])  # X-axis (microns)
    y = np.linspace(0, 2, data.shape[0])  # Y-axis (microns)
    X, Y = np.meshgrid(x, y)

    # Apply Gamma Correction
    gamma = 0.7
    gamma_corrected_img = np.power(flattened_img / 255.0, gamma) * 255.0
    gamma_corrected_img = gamma_corrected_img.astype(np.uint8)

    # Apply Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(gamma_corrected_img, 9, 30, 75)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(bilateral_filtered, (3, 3), 3)

    # Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 1
    )

    # Contour Detection
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    output_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    curves_length = 0
    circular_shapes_area = 0
    extended_shapes_area = 0
    circular_shapes = []
    curved_lines = []
    extended_shapes = []

    for contour in contours:
        if cv2.contourArea(contour) > 0:
            perimeter = cv2.arcLength(contour, closed=True)
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter != 0 else 0

            # Create a mask for the contour
            mask = np.zeros_like(thresholded, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            # Check if the contour is inside a covered area (valid contour)
            if not np.any(coverage_mask[mask == 255]):
                continue  # Skip this contour if it falls in an uncovered region

            # Calculate the total number of pixels inside the contour
            total_pixels = cv2.countNonZero(mask)

            # Calculate the number of white pixels inside the contour
            white_pixels = cv2.countNonZero(cv2.bitwise_and(mask, thresholded))
            # Calculate the number of dark pixels inside the contour (where the thresholded image is 0)
            dark_pixels = total_pixels - cv2.countNonZero(
                cv2.bitwise_and(mask, thresholded)
            )

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if circularity > 0.2 and area <= 1000:
                # Check for white pixel coverage only for circular shapes
                # if dark_pixels / total_pixels >= 0.8:
                # continue  # Skip this contour if 80% or more of the pixels are white

                # Process circular shapes
                circular_shapes.append(contour)
                circular_shapes_area += area
                cv2.drawContours(output_image, [contour], -1, (0, 255, 255), 1)
            elif aspect_ratio > 2 and area <= 1000:
                # Check for white pixel coverage only for extended shapes
                # if total_pixels > 0 and (white_pixels / total_pixels) >= 0.8:
                # continue  # Skip this contour if 80% or more of the pixels are white

                # Process extended shapes
                extended_shapes.append(contour)
                extended_shapes_area += area
                cv2.drawContours(output_image, [contour], -1, (0, 255, 255), 1)

            else:
                # First, calculate the area of the contour
                contour_area = cv2.contourArea(contour)

                if contour_area <= 250:  # Threshold for small areas
                    extended_shapes.append(contour)
                    extended_shapes_area += contour_area
                    cv2.drawContours(output_image, [contour], -1, (0, 255, 255), 1)
                else:
                    # Process curved lines without the white pixel check
                    curved_lines.append(contour)
                    curve_length = cv2.arcLength(contour, closed=True)
                    curves_length += curve_length
                    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 1)

    # Convert areas and lengths to micrometers
    circular_shapes_area_um2 = circular_shapes_area * (pixel_size_um**2)
    extended_shapes_area_um2 = extended_shapes_area * (pixel_size_um**2)
    curves_length_um = curves_length * pixel_size_um

    Total_Defect_Area = circular_shapes_area_um2 + extended_shapes_area_um2
    Total_Defect_Percentage = 100 * Total_Defect_Area / (height * width)

    results = {
        "average_current": average_current,
        "coverage_percentage": coverage_percentage,
        "circular_shapes_area": circular_shapes_area_um2,
        "extended_shapes_area": extended_shapes_area_um2,
        "curves_length": curves_length_um,
        "total_defect_area": Total_Defect_Area,
        "total_defect_percentage": Total_Defect_Percentage,
        "number_ext_shapes": len(extended_shapes),
    }

    return results


def calculate_roughness(height_data: np.ndarray) -> dict:

    if isinstance(height_data, torch.Tensor):
        height_data = height_data.cpu().numpy()

    # Zero-centering the data
    height_data -= np.mean(height_data)

    # RMS roughness in nm
    Sq = np.sqrt(np.mean(height_data**2)) * 1e9

    # Mean roughness in nm
    Sa = np.mean(np.abs(height_data)) * 1e9

    return {"RMS roughness (Sq)": Sq, "Mean roughness (Sa)": Sa}


def pcnt_abs_diff_surface_roughness(x1: torch.Tensor, x2: torch.Tensor) -> dict:

    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()

    char_1 = calculate_roughness(x1)
    char_2 = calculate_roughness(x2)

    diffs = {}
    for k in char_1:
        v1, v2 = char_1[k], char_2[k]
        err = abs(v1 - v2)
        diffs[k] = err

    return diffs


def calculate_abs_diff_between_samples(
    x1: torch.Tensor, x2: torch.Tensor, image_size_um: float
) -> dict:

    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
    char_1 = process_image(x1, x1.shape[0], x1.shape[1], image_size_um)
    char_2 = process_image(x2, x2.shape[1], x2.shape[1], image_size_um)

    diffs = {}
    for k in char_1:
        v1, v2 = char_1[k], char_2[k]
        err = abs(v1 - v2)
        diffs[k] = err

    return diffs


if __name__ == "__main__":

    dummy = np.random.random((128, 128))
    res = process_image(dummy, 128, 128)

    breakpoint()
