"""
Image dewarping utilities using Thin Plate Spline transformations.

This module contains functions for:
- Thin Plate Spline (TPS) transformations for image dewarping
- Line-based distortion analysis and correction
- Global and local dewarping strategies
- TPS control point calculation and optimization
"""

import cv2
import scipy
import numpy as np
import numpy.typing as npt
from tps import ThinPlateSpline
from typing import List

from BDRC.Data import Line


def run_tps(image: npt.NDArray, input_pts, output_pts, add_corners=True, alpha=0.5):
    """
    Apply Thin Plate Spline transformation to dewarp an image.
    
    Args:
        image: Input image to transform
        input_pts: Source control points
        output_pts: Target control points
        add_corners: Whether to add corner control points
        alpha: TPS regularization parameter
        
    Returns:
        Dewarped image array
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width, _ = image.shape

    input_pts = npt.NDArray(input_pts)
    output_pts = npt.NDArray(output_pts)

    if add_corners:
        corners = npt.NDArray(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        corners *= [height, width]
        corners *= [height, width]

        input_pts = np.concatenate((input_pts, corners))
        output_pts = np.concatenate((output_pts, corners))

    tps = ThinPlateSpline(alpha)
    tps.fit(input_pts, output_pts)

    output_indices = np.indices((height, width), dtype=np.float64).transpose(1, 2, 0)  # Shape: (H, W, 2)
    input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(height, width, 2)
    warped = np.concatenate(
        [
            scipy.ndimage.map_coordinates(image[..., channel], input_indices.transpose(2, 0, 1))[..., None]
            for channel in (0, 1, 2)
        ],
        axis=-1,
    )

    return warped


def get_global_center(slice_image: npt.NDArray, start_x: int, bbox_y: int):
    """
    Transfer coordinates of a 'local' bbox taken from a line back to the image space.
    
    Args:
        slice_image: Image slice containing line segment
        start_x: Starting x-coordinate of the slice in global space
        bbox_y: Y-coordinate of bounding box in global space
        
    Returns:
        Tuple of (global_x, global_y, bbox_height)
    """
    contours, _ = cv2.findContours(slice_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if contours is empty
    if not contours:
        # Return default values based on the slice_image dimensions
        center_x = slice_image.shape[1] // 2
        center_y = slice_image.shape[0] // 2
        bbox_h = slice_image.shape[0]
        
        global_x = start_x + center_x
        global_y = bbox_y + center_y
        
        return global_x, global_y, bbox_h
    
    areas = [cv2.contourArea(x) for x in contours]
    biggest_idx = areas.index(max(areas))
    biggest_contour = contours[biggest_idx]
    _, _, _, bbox_h = cv2.boundingRect(biggest_contour)
    center, _, _ = cv2.minAreaRect(biggest_contour)

    center_x = int(center[0])
    center_y = int(center[1])

    global_x = start_x + center_x
    global_y = bbox_y + center_y

    return global_x, global_y, bbox_h


def check_line_tps(image: npt.NDArray, contour: npt.NDArray, slice_width: int = 40):
    """
    Check if a line contour requires TPS correction by analyzing distortion.
    
    Args:
        image: Input image array
        contour: Line contour to analyze
        slice_width: Width of analysis slices
        
    Returns:
        Tuple of (needs_tps, input_points, output_points, max_y_delta)
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(contour)

    cv2.drawContours(mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1)

    slice1_start_x = x
    slice1_end_x = x+slice_width

    slice2_start_x = x+w//4-slice_width
    slice2_end_x = x+w//4

    slice3_start_x = x+w//2
    slice3_end_x = x+w//2+slice_width

    slice4_start_x = x+w//2+w//4
    slice4_end_x = x+w//2+(w//4)+slice_width

    slice5_start_x = x+w-slice_width
    slice5_end_x = x+w

    # define slices along the bbox from left to right
    slice_1 = mask[y:y+h, slice1_start_x:slice1_end_x, 0]
    slice_2 = mask[y:y+h, slice2_start_x:slice2_end_x, 0]
    slice_3 = mask[y:y+h, slice3_start_x:slice3_end_x, 0]
    slice_4 = mask[y:y+h, slice4_start_x:slice4_end_x, 0]
    slice_5 = mask[y:y+h, slice5_start_x:slice5_end_x, 0]

    slice1_center_x, slice1_center_y, bbox1_h = get_global_center(slice_1, slice1_start_x, y)
    slice2_center_x, slice2_center_y, bbox2_h = get_global_center(slice_2, slice2_start_x, y)
    slice3_center_x, slice3_center_y, bbox3_h = get_global_center(slice_3, slice3_start_x, y)
    slice4_center_x, slice4_center_y, bbox4_h = get_global_center(slice_4, slice4_start_x, y)
    slice5_center_x, slice5_center_y, bbox5_h = get_global_center(slice_5, slice5_start_x, y)

    all_bboxes = [bbox1_h, bbox2_h, bbox3_h, bbox4_h, bbox5_h]
    all_centers = [slice1_center_y, slice2_center_y, slice3_center_y, slice4_center_y, slice5_center_y]

    min_value = min(all_centers)
    max_value = max(all_centers)
    max_ydelta = max_value-min_value
    mean_bbox_h = np.mean(all_bboxes)
    mean_center_y = np.mean(all_centers)

    if max_ydelta > mean_bbox_h:
        target_y = round(mean_center_y)

        input_pts = [
            [slice1_center_y, slice1_center_x],
            [slice2_center_y, slice2_center_x],
            [slice3_center_y, slice3_center_x],
            [slice4_center_y, slice4_center_x],
            [slice5_center_y, slice5_center_x]
        ]

        output_pts = [
            [target_y, slice1_center_x],
            [target_y, slice2_center_x],
            [target_y, slice3_center_x],
            [target_y, slice4_center_x],
            [target_y, slice5_center_x]
        ]

        return True, input_pts, output_pts, max_ydelta
    else:
        return False, None, None, 0.0


def check_for_tps(image: npt.NDArray, line_contours: List[npt.NDArray]):
    """
    Check if an image requires TPS dewarping based on line distortion analysis.
    
    Args:
        image: Input image array
        line_contours: List of detected line contours
        
    Returns:
        Tuple of (tps_ratio, line_data_with_tps_info)
    """
    line_data = []
    for _, line_cnt in enumerate(line_contours):
        _, y, _, _ = cv2.boundingRect(line_cnt)
        # TODO: store input and output points to avoid running that step twice
        tps_status, input_pts, output_pts, max_yd = check_line_tps(image, line_cnt)

        line = {
            "contour": line_cnt,
            "tps": tps_status,
            "input_pts": input_pts,
            "output_pts": output_pts,
            "max_yd": max_yd
        }

        line_data.append(line)

    do_tps = [x["tps"] for x in line_data if x["tps"] is True]
    ratio = len(do_tps) / len(line_contours)

    return ratio, line_data


def get_global_tps_line(line_data: List):
    """
    Find the most representative curved line for global TPS transformation.
    
    A simple approach to the most representative curved line in the image assuming 
    that the overall distortion is relatively uniform.
    
    Args:
        line_data: List of line analysis data with TPS information
        
    Returns:
        Index of the best line for global TPS transformation
    """
    all_y_deltas = []

    for line in line_data:
        if line["tps"] is True:
            all_y_deltas.append(line["max_yd"])
        else:
            all_y_deltas.append(0.0)

    mean_delta = np.mean(all_y_deltas)
    best_diff = max(all_y_deltas) # just setting it to the highest value
    best_y = None

    for yd in all_y_deltas:
        if yd > 0:
            delta = abs(mean_delta - yd)
            if delta < best_diff:
                best_diff = delta
                best_y = yd

    target_idx = all_y_deltas.index(best_y)

    return target_idx


def apply_global_tps(image: npt.NDArray, line_mask: npt.NDArray, line_data: List):
    """
    Apply global TPS transformation to dewarp an entire image.
    
    Args:
        image: Input image array
        line_mask: Binary mask of detected lines
        line_data: List of line analysis data with TPS information
        
    Returns:
        Tuple of (dewarped_image, dewarped_mask)
    """
    best_idx = get_global_tps_line(line_data)
    output_pts = line_data[best_idx]["output_pts"]
    input_pts = line_data[best_idx]["input_pts"]

    assert input_pts is not None and output_pts is not None

    warped_img = run_tps(image, output_pts, input_pts)
    warped_mask = run_tps(line_mask, output_pts, input_pts)

    return warped_img, warped_mask


def get_line_images_via_local_tps(image: npt.NDArray, line_data: list, k_factor: float = 1.7):
    """
    Extract line images using local TPS transformations for each line.
    
    Args:
        image: Input image array
        line_data: List of line analysis data with TPS information
        k_factor: Scaling factor for line extraction
        
    Returns:
        List of extracted and dewarped line images
    """
    # Import get_line_image locally to avoid circular dependency
    from BDRC.line_detection import get_line_image
    
    default_k_factor = k_factor
    current_k = default_k_factor
    line_images = []

    for line in line_data:
        if line["tps"] is True:
            output_pts = line["output_pts"]
            input_pts = line["input_pts"]

            assert input_pts is not None and output_pts is not None

            tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(tmp_mask, [line["contour"]], -1, (255, 255, 255), -1)

            warped_img = run_tps(image, output_pts, input_pts)
            warped_mask = run_tps(tmp_mask, output_pts, input_pts)

            _, _, _, bbox_h = cv2.boundingRect(line["contour"])

            line_img, adapted_k = get_line_image(warped_img, warped_mask, bbox_h, bbox_tolerance=2.0,
                                                 k_factor=current_k)
            line_images.append(line_img)

            if current_k != adapted_k:
                current_k = adapted_k

        else:
            tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(tmp_mask, [line["contour"]], -1, (255, 255, 255), -1)

            _, _, _, h = cv2.boundingRect(line["contour"])
            line_img, adapted_k = get_line_image(image, tmp_mask, h, bbox_tolerance=2.0, k_factor=current_k)
            line_images.append(line_img)

    return line_images