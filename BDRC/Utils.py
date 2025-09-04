"""
Utility functions for image processing, OCR operations, and file management.

This module contains a comprehensive set of utility functions for:
- Image processing and transformation
- Screen and platform detection
- OCR model management
- File operations and directory handling
- Line detection and sorting algorithms
- Image dewarping and geometric transformations
"""

import os
import cv2
import json
import math
import scipy
import logging
import platform
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from math import ceil
from uuid import uuid1
from pathlib import Path
from datetime import datetime
from tps import ThinPlateSpline
from typing import List, Tuple, Optional, Sequence

from BDRC.Data import OCRModelConfig, Platform, ScreenData, BBox, Line, \
    OCRModel, OCRData
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, Qt

from Config import OCRARCHITECTURE, CHARSETENCODER

page_classes = {
                "background": "0, 0, 0",
                "image": "45, 255, 0",
                "line": "255, 100, 0",
                "margin": "255, 0, 0",
                "caption": "255, 100, 243"
            }

def get_screen_center(app: QApplication, start_size_ratio: float = 0.8) -> ScreenData:
    """
    Calculate optimal window size and position for the application.
    
    Args:
        app: QApplication instance to get screen information from
        start_size_ratio: Ratio of screen size to use for initial window size
        
    Returns:
        ScreenData with calculated dimensions and positioning
    """
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    max_width = rect.width()
    max_height = rect.height()

    start_width = int(rect.width() * start_size_ratio)
    start_height = int(rect.height() * start_size_ratio)

    start_pos_x = (max_width - start_width) // 2
    start_pos_y = (max_height - start_height) // 2

    screen_data = ScreenData(
        max_width=max_width,
        max_height=max_height,
        start_width=start_width,
        start_height=start_height,
        start_x=start_pos_x,
        start_y=start_pos_y,
    )

    return screen_data


def get_platform() -> Platform:
    """
    Detect the current operating system platform.
    
    Returns:
        Platform enum value for Windows, Mac, or Linux
    """
    _platform_tag = platform.platform()
    _platform_tag = _platform_tag.split("-")[0]

    if _platform_tag == "Windows":
        _platform = Platform.Windows
    elif _platform_tag == "macOS":
        _platform = Platform.Mac
    else:
        _platform = Platform.Linux

    return _platform


def get_utc_time():
    """
    Get current UTC time as a formatted string.
    
    Returns:
        Current UTC time in ISO format (YYYY-MM-DDTHH:MM:SS)
    """
    utc_time = datetime.now()
    utc_time = utc_time.strftime('%Y-%m-%dT%H:%M:%S')

    return utc_time

def get_execution_providers() -> List[str]:
    """
    Get available ONNX runtime execution providers.
    
    Returns:
        List of available execution provider names
    """
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    return available_providers

def get_filename(file_path: str) -> str:
    """
    Extract filename without extension from a file path.
    
    Args:
        file_path: Full path to the file
        
    Returns:
        Filename without extension
    """
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def create_dir(dir_name: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        dir_name: Path of the directory to create
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"Created directory at  {dir_name}")
        except IOError as e:
            print(f"Failed to create directory at: {dir_name}, {e}")


def generate_guid(clock_seq: int):
    """
    Generate a UUID with a specific clock sequence.
    
    Args:
        clock_seq: Clock sequence value for UUID generation
        
    Returns:
        Generated UUID
    """
    return uuid1(clock_seq=clock_seq)


def build_ocr_data(id_val, file_path: str, target_width: int = None):
    """
    Build OCR data from a file path.
    
    Args:
        id_val: Either an integer or a UUID to use as the identifier
        file_path: Path to the image file
        target_width: Optional width to scale the image to
    
    Returns:
        OCRData object
    """
    file_name = get_filename(file_path)
    
    # Generate GUID if id_val is an integer, otherwise use the provided UUID
    if isinstance(id_val, int):
        guid = generate_guid(id_val)
    else:
        guid = id_val
    
    # Load and scale the image
    if target_width is not None:
        q_image = QImage(file_path).scaledToWidth(target_width, Qt.TransformationMode.SmoothTransformation)
    else:
        q_image = QImage(file_path)
    
    ocr_data = OCRData(
        guid=guid,
        image_path=file_path,
        image_name=file_name,
        qimage=q_image,
        ocr_lines=None,
        lines=None,
        preview=None,
        angle=0.0
    )

    return ocr_data

def read_theme_file(file_path: str) -> dict | None:
    """
    Load theme configuration from a JSON file.
    
    Args:
        file_path: Path to the theme configuration file
        
    Returns:
        Theme configuration dictionary, or None if file doesn't exist
    """
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            content = json.load(f)

        return content
    else:
        logging.error(f"Theme File {file_path} does not exist")
        return None


def import_local_models(model_path: str):
    """
    Import all OCR models from a directory.
    
    Args:
        model_path: Directory path containing OCR model subdirectories
        
    Returns:
        List of OCRModel instances loaded from the directory
    """
    tick = 1
    ocr_models = []

    if os.path.isdir(model_path):
        for sub_dir in Path(model_path).iterdir():
            if os.path.isdir(sub_dir):
                _config_file = os.path.join(sub_dir, "model_config.json")
                if not os.path.isfile(_config_file):
                    logging.warn("ignore "+str(sub_dir))
                    tick += 1
                    continue

                _config = read_ocr_model_config(_config_file)
                _model = OCRModel(
                    guid=generate_guid(tick),
                    name=sub_dir.name,
                    path=str(sub_dir),
                    config=_config
                )
                ocr_models.append(_model)
            tick += 1

    return ocr_models

def import_local_model(model_path: str):
    """
    Import a single OCR model from a directory.
    
    Args:
        model_path: Directory path containing a single OCR model
        
    Returns:
        OCRModel instance or None if model cannot be loaded
    """
    _model = None
    if os.path.isdir(model_path):
        _config_file = os.path.join(model_path, "model_config.json")
        if not os.path.isfile(_config_file):
            return None

        _config = read_ocr_model_config(_config_file)
        _model = OCRModel(
            guid=generate_guid(1),
            name=Path(model_path).name,
            path=model_path,
            config=_config
        )

    return _model

def read_ocr_model_config(config_file: str):
    """
    Load OCR model configuration from a JSON file.
    
    Args:
        config_file: Path to the model configuration JSON file
        
    Returns:
        OCRModelConfig instance with loaded parameters
    """
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    version = json_content["version"]
    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    encoder = json_content["encoder"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = json_content["charset"]
    add_blank = True if json_content["add_blank"] == "yes" else False

    config = OCRModelConfig(
        onnx_model_file,
        OCRARCHITECTURE[architecture],
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        encoder=CHARSETENCODER[encoder],
        charset=characters,
        add_blank=add_blank,
        version=version
    )

    return config


def resize_to_height(image, target_height: int):
    """
    Resize image to a specific height while maintaining aspect ratio.
    
    Args:
        image: Input image array
        target_height: Desired height in pixels
        
    Returns:
        Tuple of (resized_image, scale_ratio)
    """
    scale_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * scale_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def resize_to_width(image, target_width: int = 2048):
    """
    Resize image to a specific width while maintaining aspect ratio.
    
    Args:
        image: Input image array
        target_width: Desired width in pixels (default: 2048)
        
    Returns:
        Tuple of (resized_image, scale_ratio)
    """
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio

def calculate_steps(image: npt.NDArray, patch_size: int = 512) -> Tuple[int, int]:
    """
    Calculate number of patches needed to tile an image.
    
    Args:
        image: Input image array
        patch_size: Size of each square patch (default: 512)
        
    Returns:
        Tuple of (x_steps, y_steps) for tiling
    """
    x_steps = image.shape[1] / patch_size
    y_steps = image.shape[0] / patch_size

    x_steps = math.ceil(x_steps)
    y_steps = math.ceil(y_steps)

    return x_steps, y_steps


def calculate_paddings(
    image: npt.NDArray, x_steps: int, y_steps: int, patch_size: int = 512
) -> tuple[int, int]:
    """
    Calculate padding needed to make image divisible into patches.
    
    Args:
        image: Input image array
        x_steps: Number of horizontal patches
        y_steps: Number of vertical patches
        patch_size: Size of each patch
        
    Returns:
        Tuple of (pad_x, pad_y) padding values
    """
    max_x = x_steps * patch_size
    max_y = y_steps * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def pad_image(
    image: npt.NDArray, pad_x: int, pad_y: int, pad_value: int = 0
) -> npt.NDArray:
    """
    Add padding to an image.
    
    Args:
        image: Input image array
        pad_x: Horizontal padding to add
        pad_y: Vertical padding to add
        pad_value: Value to use for padding (default: 0)
        
    Returns:
        Padded image array
    """
    padded_img = np.pad(
        image,
        pad_width=((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img

def get_contours(image: npt.NDArray) -> Sequence:
    """
    Find contours in a binary image.
    
    Args:
        image: Binary image array
        
    Returns:
        Sequence of detected contours
    """
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sigmoid(x):
    """
    Apply sigmoid activation function.
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid of input (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-x))


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

def get_line_images_via_local_tps(image: npt.NDArray, line_data: list, k_factor: float = 1.7):
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


def get_text_area(
    image: np.array, prediction: npt.NDArray
) -> Tuple[np.array, BBox] | Tuple[None, None, None]:
    dil_kernel = np.ones((12, 2))
    dil_prediction = cv2.dilate(prediction, kernel=dil_kernel, iterations=10)

    prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    dil_prediction = cv2.resize(dil_prediction, (image.shape[1], image.shape[0]))

    contours, _ = cv2.findContours(dil_prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        area_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

        area_sizes = [cv2.contourArea(x) for x in contours]
        biggest_area = max(area_sizes)
        biggest_idx = area_sizes.index(biggest_area)

        x, y, w, h = cv2.boundingRect(contours[biggest_idx])
        color = (255, 255, 255)

        cv2.rectangle(
            area_mask,
            (x, y),
            (x + w, y + h),
            color,
            -1,
        )
        area_mask = cv2.cvtColor(area_mask, cv2.COLOR_BGR2GRAY)

        return prediction, area_mask, contours[biggest_idx]
    else:
        return None, None, None

def get_text_bbox(lines: List[Line]):
    all_bboxes = [x.bbox for x in lines]
    min_x = min(a.x for a in all_bboxes)
    min_y = min(a.y for a in all_bboxes)

    max_w = max(a.w for a in all_bboxes)
    max_h = all_bboxes[-1].y + all_bboxes[-1].h

    bbox = BBox(min_x, min_y, max_w, max_h)

    return bbox

def mask_n_crop(image: np.array, mask: np.array) -> np.array:
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    image_masked = cv2.bitwise_and(image, image, mask, mask)
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
    )
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=0))[0], axis=1
    )

    return image_masked

def calculate_rotation_angle_from_lines(
    line_mask: npt.NDArray,
    max_angle: float = 5.0,
    debug_angles: bool = False,
) -> float:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    
    # Check if contours is empty before proceeding
    if not contours:
        return 0.0
        
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if debug_angles:
        print(f"All Angles: {angles}")

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = np.mean(low_angles)

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = -(90 - np.mean(high_angles))

    else:
        mean_angle = 0.0

    return mean_angle


def rotate_from_angle(image: np.array, angle: float) -> np.array:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img


def get_rotation_angle_from_lines(
    line_mask: npt.NDArray,
    max_angle: float = 5.0,
    debug_angles: bool = False,
) -> float:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if debug_angles:
        print(f"All Angles: {angles}")

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = np.mean(low_angles)

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = -(90 - np.mean(high_angles))

    else:
        mean_angle = 0

    return mean_angle


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, center: Tuple[int, int], angle: float):
    cx = center[0]
    cy = center[1]

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def is_inside_rectangle(point, rect):
    x, y = point
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax

def filter_contours(prediction: np.array, textarea_contour: np.array) -> list[np.array]:
    filtered_contours = []
    x, y, w, h = cv2.boundingRect(textarea_contour)
    line_contours, _ = cv2.findContours(
        prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in line_contours:
        center, _, angle = cv2.minAreaRect(cnt)
        is_in_area = is_inside_rectangle(center, [x, y, x + w, y + h])

        if is_in_area:
            filtered_contours.append(cnt)

    return filtered_contours


def post_process_prediction(image: np.array, prediction: np.array):
    prediction, text_area, textarea_contour = get_text_area(image, prediction)

    if prediction is not None:
        cropped_prediction = mask_n_crop(prediction, text_area)
        angle = calculate_rotation_angle_from_lines(cropped_prediction)

        rotated_image = rotate_from_angle(image, angle)
        rotated_prediction = rotate_from_angle(prediction, angle)

        M = cv2.moments(textarea_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        rotated_textarea_contour = rotate_contour(textarea_contour, (cx, cy), angle)

        return rotated_image, rotated_prediction, rotated_textarea_contour, angle
    else:
        return None, None, None, None


def generate_line_preview(prediction: np.array, filtered_contours: list[np.array]):
    preview = np.zeros(shape=prediction.shape, dtype=np.uint8)

    for cnt in filtered_contours:
        cv2.drawContours(preview, [cnt], -1, color=(255, 0, 0), thickness=-1)

    return preview


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)



def build_line_data(contour: np.array, optimize: bool = True) -> Line:
    """
    Create a Line object from a contour with bounding box and center information.
    
    Args:
        contour: Contour points array
        optimize: Whether to optimize the contour shape
        
    Returns:
        Line object with GUID, contour, bounding box, and center point
    """
    if optimize:
        contour = optimize_countour(contour)

    x, y, w, h = cv2.boundingRect(contour)

    x_center = x + (w // 2)
    y_center = y + (h // 2)

    bbox = BBox(x, y, w, h)
    guid = generate_guid(clock_seq=23)
    return Line(guid, contour, bbox, (x_center, y_center))


def get_line_threshold(line_prediction: npt.NDArray, slice_width: int = 20):
    """
    This function generates n slices (of n = steps) width the width of slice_width across the bbox of the detected lines.
    The slice with the max. number of contained contours is taken to be the canditate to calculate the bbox center of each contour and
    take the median distance between each bbox center as estimated line cut-off threshold to sort each line segment across the horizontal

    Note: This approach might turn out to be problematic in case of sparsely spread line segments across a page
    """

    if len(line_prediction.shape) == 3:
        line_prediction = cv2.cvtColor(line_prediction, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(line_prediction)
    x_steps = (w // slice_width) // 2

    bbox_numbers = []

    for step in range(1, x_steps + 1):
        x_offset = x_steps * step
        x_start = x + x_offset
        x_end = x_start + slice_width

        _slice = line_prediction[y : y + h, x_start:x_end]
        contours, _ = cv2.findContours(_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bbox_numbers.append((len(contours), contours))

    sorted_list = sorted(bbox_numbers, key=lambda x: x[0], reverse=True)

    if len(sorted_list) > 0:
        reference_slice = sorted_list[0]

        y_points = []
        n_contours, contours = reference_slice

        if n_contours == 0:
            print("number of contours is 0")
            line_threshold = 0.0
        else:
            for _, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                y_center = y + (h // 2)
                y_points.append(y_center)

            # Check if y_points is empty before calculating median
            if len(y_points) > 0:
                line_threshold = float(np.median(y_points) // n_contours)
            else:
                line_threshold = 0.0
    else:
        line_threshold = 0.0

    return line_threshold


def sort_bbox_centers(bbox_centers: List[Tuple[int, int]], line_threshold: int = 20) -> List:
    # Handle empty bbox_centers
    if not bbox_centers:
        return []
        
    sorted_bbox_centers = []
    tmp_line = []

    for i in range(0, len(bbox_centers)):
        if len(tmp_line) > 0:
            for s in range(0, len(tmp_line)):

                # TODO: refactor this to make this calculation an enum to choose between both methods
                # y_diff = abs(tmp_line[s][1] - bbox_centers[i][1])
                """
                I use the mean of the hitherto present line chunks in tmp_line since
                the precalculated fixed threshold can break the sorting if
                there is some slight bending in the line. This part may need some tweaking after
                some further practical review
                """
                ys = [y[1] for y in tmp_line]
                
                # Check if ys is not empty before calculating mean
                if ys:
                    mean_y = np.mean(ys)
                    y_diff = abs(mean_y - bbox_centers[i][1])

                    if y_diff > line_threshold:
                        tmp_line.sort(key=lambda x: x[0])
                        sorted_bbox_centers.append(tmp_line.copy())
                        tmp_line.clear()

                        tmp_line.append(bbox_centers[i])
                        break
                    else:
                        tmp_line.append(bbox_centers[i])
                        break
                else:
                    tmp_line.append(bbox_centers[i])
                    break
        else:
            tmp_line.append(bbox_centers[i])

    # Add the last tmp_line if it's not empty
    if tmp_line:
        sorted_bbox_centers.append(tmp_line)

    # Sort each line by x-coordinate
    for y in sorted_bbox_centers:
        y.sort(key=lambda x: x[0])

    sorted_bbox_centers = list(reversed(sorted_bbox_centers))

    return sorted_bbox_centers


def group_line_chunks(sorted_bbox_centers, lines: List[Line], adaptive_grouping: bool = True):
    new_line_data = []
    for bbox_centers in sorted_bbox_centers:

        if len(bbox_centers) > 1:  # i.e. more than 1 bbox center in a group
            contour_stack = []

            for box_center in bbox_centers:
                for line_data in lines:
                    if box_center == line_data.center:
                        contour_stack.append(line_data.contour)
                        break

            if adaptive_grouping:
                for contour in contour_stack:
                    x, y, w, h = cv2.boundingRect(contour)
                    width_offset = int(w * 0.05)
                    height_offset = int(h * 0.05)
                    w += width_offset
                    h += height_offset

            stacked_contour = np.vstack(contour_stack)
            stacked_contour = cv2.convexHull(stacked_contour)

            # TODO: both calls necessary?
            x, y, w, h = cv2.boundingRect(stacked_contour)
            _, _, angle = cv2.minAreaRect(stacked_contour)

            _bbox = BBox(x, y, w, h)
            x_center = _bbox.x + (_bbox.w // 2)
            y_center = _bbox.y + (_bbox.h // 2)

            new_line = Line(
                guid=generate_guid(clock_seq=23),
                contour=stacked_contour,
                bbox=_bbox,
                center=(x_center, y_center)
            )

            new_line_data.append(new_line)

        else:
            for _bcenter in bbox_centers:
                for line_data in lines:
                    if _bcenter == line_data.center:
                        new_line_data.append(line_data)
                        break

    return new_line_data


def sort_lines_by_threshold(
    line_mask: np.array,
    lines: list[Line],
    threshold: int = 20,
    calculate_threshold: bool = True,
    group_lines: bool = True
):
    """
    Sort detected lines by vertical position using threshold-based grouping.
    
    Args:
        line_mask: Binary mask of detected lines
        lines: List of Line objects to sort
        threshold: Vertical distance threshold for grouping
        calculate_threshold: Whether to auto-calculate threshold
        group_lines: Whether to merge nearby lines
        
    Returns:
        Tuple of (sorted_lines, calculated_threshold)
    """
    bbox_centers = [x.center for x in lines]

    if calculate_threshold:
        line_treshold = get_line_threshold(line_mask)
    else:
        line_treshold = threshold

    sorted_bbox_centers = sort_bbox_centers(bbox_centers, line_threshold=line_treshold)

    if group_lines:
        new_lines = group_line_chunks(sorted_bbox_centers, lines)
    else:
        _bboxes = [x for xs in sorted_bbox_centers for x in xs]

        new_lines = []
        for _bbox in _bboxes:
            for _line in lines:
                if _bbox == _line.center:
                    new_lines.append(_line)

    return new_lines, line_treshold


def sort_lines_by_threshold2(
    line_mask: npt.NDArray,
    lines: List[Line],
    threshold: int = 20,
    calculate_threshold: bool = True,
    group_lines: bool = True
):

    bbox_centers = [x.center for x in lines]

    if calculate_threshold:
        line_treshold = get_line_threshold(line_mask)
    else:
        line_treshold = threshold

    sorted_bbox_centers = sort_bbox_centers(bbox_centers, line_threshold=line_treshold)

    if group_lines:
        new_lines = group_line_chunks(sorted_bbox_centers, lines)
    else:
        _bboxes = [x for xs in sorted_bbox_centers for x in xs]

        new_lines = []
        for _bbox in _bboxes:
            for _line in lines:
                if _bbox == _line.center:
                    new_lines.append(_line)

    return new_lines, line_treshold


def build_raw_line_data(image: npt.NDArray, line_mask: npt.NDArray):

    if len(line_mask.shape) == 3:
        line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)

    angle = get_rotation_angle_from_lines(line_mask)
    rot_mask = rotate_from_angle(line_mask, angle)
    rot_img = rotate_from_angle(image, angle)

    line_contours = get_contours(rot_mask)
    line_contours = [x for x in line_contours if cv2.contourArea(x) > 10]

    rot_mask = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2RGB)

    return rot_img, rot_mask, line_contours, angle

def tile_image(padded_img: npt.NDArray, patch_size: int = 512):
    x_steps = int(padded_img.shape[1] / patch_size)
    y_steps = int(padded_img.shape[0] / patch_size)
    y_splits = np.split(padded_img, y_steps, axis=0)

    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    patches = [x for xs in patches for x in xs]

    return patches, y_steps

def stitch_predictions(prediction: npt.NDArray, y_steps: int) -> npt.NDArray:
    pred_y_split = np.split(prediction, y_steps, axis=0)
    x_slices = [np.hstack(x) for x in pred_y_split]
    concat_img = np.vstack(x_slices)

    return concat_img


def get_paddings(image: npt.NDArray, patch_size: int = 512) -> Tuple[int, int]:
    max_x = ceil(image.shape[1] / patch_size) * patch_size
    max_y = ceil(image.shape[0] / patch_size) * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def preprocess_image(
    image: npt.NDArray,
    patch_size: int = 512,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
    clamp_size: bool = True,
):
    """
    Preprocess image for OCR by resizing and padding to patch-compatible dimensions.
    
    Some dimension checking and resizing to avoid very large inputs on which the line(s) 
    on the resulting tiles could be too big and cause troubles with the current line model.
    
    Args:
        image: Input image array
        patch_size: Target patch size for tiling
        clamp_width: Maximum allowed width
        clamp_height: Maximum allowed height
        clamp_size: Whether to enforce size limits
        
    Returns:
        Tuple of (processed_image, pad_x, pad_y)
    """
    if clamp_size and image.shape[1] > image.shape[0] and image.shape[1] > clamp_width:
        image, _ = resize_to_width(image, clamp_width)

    elif (
        clamp_size and image.shape[0] > image.shape[1] and image.shape[0] > clamp_height
    ):
        image, _ = resize_to_height(image, clamp_height)

    elif image.shape[0] < patch_size:
        image, _ = resize_to_height(image, patch_size)

    pad_x, pad_y = get_paddings(image, patch_size)
    padded_img = pad_image(image, pad_x, pad_y, pad_value=255)

    return padded_img, pad_x, pad_y

def filter_line_contours(image: npt.NDArray, line_contours, threshold: float = 0.01) -> List:
    filtered_contours = []
    for _, line_cnt in enumerate(line_contours):

        _, _, w, h = cv2.boundingRect(line_cnt)

        if w > image.shape[1] * threshold and h > 10:
            filtered_contours.append(line_cnt)

    return filtered_contours


def extract_line(image: npt.NDArray, mask: npt.NDArray, bbox_h: int, k_factor: float = 1.2):
    k_size = int(bbox_h * k_factor)
    morph_multiplier = k_factor

    morph_rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(k_size, int(k_size * morph_multiplier)))
    iterations = 1
    dilated_mask = cv2.dilate(mask, kernel=morph_rect, iterations=iterations)
    masked_line = mask_n_crop(image, dilated_mask)

    return masked_line


def get_line_image(image: npt.NDArray, mask: npt.NDArray, bbox_h: int, bbox_tolerance: float = 2.5, k_factor: float = 1.2):
    try:
        tmp_k = k_factor
        line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)
        
        # Add a safety check to prevent infinite loop
        max_attempts = 10
        attempts = 0
        
        while line_img.shape[0] > bbox_h * bbox_tolerance and attempts < max_attempts:
            tmp_k = tmp_k - 0.1
            if tmp_k <= 0.1:  # Prevent k_factor from becoming too small
                break
            line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)
            attempts += 1

        return line_img, tmp_k
    except Exception as e:
        # Return a minimal valid image and the original k_factor in case of error
        print(f"Error in get_line_image: {e}")
        # Create a small blank image as fallback
        fallback_img = np.zeros((bbox_h, bbox_h * 2, 3), dtype=np.uint8)
        return fallback_img, k_factor

def normalize(image: npt.NDArray) -> npt.NDArray:
    """
    Normalize image pixel values to range [0, 1].
    
    Args:
        image: Input image array with values in range [0, 255]
        
    Returns:
        Normalized image array with values in range [0, 1]
    """
    image = image.astype(np.float32)
    image /= 255.0
    return image


def binarize(
        img: npt.NDArray, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> npt.NDArray:
    line_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw

def pad_to_width(img: np.array, target_width: int, target_height: int, padding: str) -> np.array:
    _, width, channels = img.shape
    tmp_img, ratio = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.ones(shape=(target_height - height - middle, target_width, channels), dtype=np.uint8)

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.zeros(shape=(target_height - height - middle, target_width, channels), dtype=np.uint8)

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(img: npt.NDArray, target_width: int, target_height: int, padding: str) -> npt.NDArray:
    height, _, channels = img.shape
    tmp_img, ratio = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.ones(shape=(target_height, target_width - width - middle, channels), dtype=np.uint8)

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.zeros(shape=(target_height, target_width - width - middle, channels), dtype=np.uint8)

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img


def pad_ocr_line(
        img: npt.NDArray,
        target_width: int = 3000,
        target_height: int = 80,
        padding: str = "black") -> npt.NDArray:

    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:
        out_img = pad_to_width(img, target_width, target_height, padding)

    elif width_ratio > height_ratio:
        out_img = pad_to_height(img, target_width, target_height, padding)
    else:
        out_img = pad_to_width(img, target_width, target_height, padding)

    return cv2.resize(out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def create_preview_image(
            image: npt.NDArray,
            image_predictions: Optional[List],
            line_predictions: Optional[List],
            caption_predictions: Optional[List],
            margin_predictions: Optional[List],
            alpha: float = 0.4,
    ) -> npt.NDArray:
        mask = np.zeros(image.shape, dtype=np.uint8)

        if image_predictions is not None and len(image_predictions) > 0:
            color = tuple([int(x) for x in page_classes["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if line_predictions is not None:
            color = tuple([int(x) for x in page_classes["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in page_classes["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in page_classes["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image


def get_global_tps_line(line_data: List):
    """
    A simple approach to the most representative curved line in the image assuming that the overall distortion is relatively uniform
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


def get_global_center(slice_image: npt.NDArray, start_x: int, bbox_y: int):
    """
    Transfers the coordinates of a 'local' bbox taken from a line back to the image space
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


def apply_global_tps(image: npt.NDArray, line_mask: npt.NDArray, line_data: List):
    best_idx = get_global_tps_line(line_data)
    output_pts = line_data[best_idx]["output_pts"]
    input_pts = line_data[best_idx]["input_pts"]

    assert input_pts is not None and output_pts is not None

    warped_img = run_tps(image, output_pts, input_pts)
    warped_mask = run_tps(line_mask, output_pts, input_pts)

    return warped_img, warped_mask


def check_line_tps(image: npt.NDArray, contour: npt.NDArray, slice_width: int = 40):

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

def extract_line_images(image: npt.NDArray, line_data: List[Line], default_k: float = 1.7, bbox_tolerance: float = 3):
    default_k_factor = default_k
    current_k = default_k_factor

    line_images = []

    for _, line in enumerate(line_data):
        _, _, _, h = cv2.boundingRect(line.contour)
        tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(tmp_mask, [line.contour], -1, (255, 255, 255), -1)

        line_img, adapted_k = get_line_image(image, tmp_mask, h, bbox_tolerance=bbox_tolerance, k_factor=current_k)
        line_images.append(line_img)

        if current_k != adapted_k:
            current_k = adapted_k

    return line_images
