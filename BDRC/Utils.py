"""
Utility functions for image processing, OCR operations, and file management.

This module contains a comprehensive set of utility functions for:
- Image processing and transformation
- Screen and platform detection
- OCR model management
- File operations and directory handling

Note: Line detection/sorting and image dewarping functions have been moved to
separate modules and are re-exported here for backward compatibility.
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

# Import functions from specialized modules for backward compatibility
from BDRC.line_detection import (
    get_line_threshold,
    sort_bbox_centers,
    group_line_chunks,
    sort_lines_by_threshold,
    sort_lines_by_threshold2,
    build_line_data,
    build_raw_line_data,
    extract_line_images,
    extract_line,
    get_line_image,
    filter_line_contours,
    get_rotation_angle_from_lines,
    calculate_rotation_angle_from_lines,
    optimize_countour,
    get_contours,
    rotate_from_angle,
    mask_n_crop
)

from BDRC.image_dewarping import (
    run_tps,
    apply_global_tps,
    check_line_tps,
    check_for_tps,
    get_global_tps_line,
    get_line_images_via_local_tps,
    get_global_center
)

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

# Import generate_guid from line_detection module for backward compatibility
from BDRC.line_detection import generate_guid

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

def sigmoid(x):
    """
    Apply sigmoid activation function.
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid of input (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-x))

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

