"""
Data structures and enumerations for the Tibetan OCR application.

This module contains all the core data types, enums, and dataclasses used
throughout the OCR application for representing OCR data, settings, and
various configuration options.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from uuid import UUID

import numpy.typing as npt
from PySide6.QtGui import QImage


class OpStatus(Enum):
    """Operation status indicators for various OCR operations."""

    SUCCESS = 0
    FAILED = 1


class Platform(Enum):
    """Supported operating system platforms."""

    WINDOWS = 0
    OSX = 1
    LINUX = 2


class Encoding(Enum):
    """Text encoding formats for OCR output."""

    UNICODE = 0
    WYLIE = 1


class CharsetEncoder(Enum):
    """Character set encoding methods for OCR models."""

    WYLIE = 0
    STACK = 1


class ExportFormat(Enum):
    """Available export formats for OCR results."""

    TXT = 0
    XML = 1
    JSON = 2


class Theme(Enum):
    """Application UI theme options."""

    DARK = 0
    LIGHT = 1


class LineMode(Enum):
    """Line detection modes for OCR processing."""

    LINE = 0
    LAYOUT = 1


class LineMerge(Enum):
    """Methods for merging detected lines."""

    MERGE = 0
    STACK = 1


class LineSorting(Enum):
    """Algorithms for sorting detected lines."""

    THRESHOLD = 0
    PEAKS = 1


class OCRArchitecture(Enum):
    """Supported OCR model architectures."""

    EASTER2 = 0
    CRNN = 1


class TPSMode(Enum):
    """Thin Plate Spline transformation modes for dewarping."""

    GLOBAL = 0
    LOCAL = 1


class Language(Enum):
    """Supported application languages."""

    ENGLISH = 0
    GERMAN = 1
    FRENCH = 2
    TIBETAN = 3
    CHINESE = 4


@dataclass
class ScreenData:
    """Screen dimensions and positioning data for application window."""

    max_width: int
    max_height: int
    start_width: int
    start_height: int
    start_x: int
    start_y: int


@dataclass
class BBox:
    """Bounding box coordinates for rectangular regions."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Line:
    """Detected text line with contour and bounding box information."""

    guid: UUID
    contour: npt.NDArray
    bbox: BBox
    center: Tuple[int, int]


@dataclass
class OCRLine:
    """OCR-recognized text line with encoding information."""

    guid: UUID
    text: str
    encoding: Encoding


@dataclass
class LayoutData:
    """Layout analysis results containing detected regions and predictions."""

    image: npt.NDArray
    rotation: float
    images: List[BBox]
    text_bboxes: List[BBox]
    lines: List[Line]
    captions: List[BBox]
    margins: List[BBox]
    predictions: Dict[str, npt.NDArray]


# NOT IMPLEMENTED
@dataclass
class ThemeData:
    """UI theme configuration data (not currently implemented)."""

    name: str
    new_button: str
    import_button: str
    save_button: str
    run_button: str
    settings_button: str


@dataclass
class OCRData:
    """Complete OCR data for a single image including results and metadata."""

    guid: UUID
    image_path: str
    image_name: str
    qimage: QImage
    ocr_lines: List[OCRLine] | None
    lines: List[Line] | None
    preview: npt.NDArray | None
    angle: float


@dataclass
class LineDetectionConfig:
    """Configuration for line detection model."""

    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    """Configuration for layout detection model."""

    model_file: str
    patch_size: int
    classes: List[str]


@dataclass
class OCRModelConfig:
    """Configuration parameters for OCR model."""

    model_file: str
    architecture: OCRArchitecture
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    encoder: CharsetEncoder
    charset: List[str]
    add_blank: bool
    version: str


@dataclass
class LineDataResult:
    """Result container for line detection operations."""

    guid: UUID
    lines: List[Line]


@dataclass
class OCRLineUpdate:
    """Update message for modifying an OCR text line."""

    page_guid: UUID
    ocr_line: OCRLine


@dataclass
class OCRLineEncodingUpdate:
    """Update message for changing encoding of multiple OCR lines."""

    page_guid: UUID
    ocr_lines: List[OCRLine]


@dataclass
class OCResult:
    """Complete OCR processing result for an image."""

    guid: UUID
    mask: npt.NDArray
    lines: List[Line]
    text: List[OCRLine]
    angle: float


@dataclass
class OCRSample:
    """OCR sample data with metadata for batch processing."""

    cnt: int
    guid: UUID
    name: str
    result: OCResult


@dataclass
class OCRModel:
    """OCR model information and configuration."""

    guid: UUID
    name: str
    path: str
    config: OCRModelConfig


@dataclass
class OCRSettings:
    """User-configurable OCR processing settings."""

    line_mode: LineMode
    line_merge: LineMerge
    line_sorting: LineSorting
    k_factor: float
    bbox_tolerance: float
    dewarping: bool
    merge_lines: bool
    tps_mode: TPSMode
    output_encoding: Encoding


@dataclass
class AppSettings:
    """General application settings and preferences."""

    model_path: str
    language: Language
    encoding: Encoding
    theme: Theme


@dataclass
class ArtifactConfig:
    """Configuration for artifact saving behavior."""

    enabled: bool = True
    granularity: str = "standard"  # "minimal", "standard"
    save_detection: bool = True
    save_dewarping: bool = True
