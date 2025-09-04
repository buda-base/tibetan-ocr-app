"""
Data structures and enumerations for the Tibetan OCR application.

This module contains all the core data types, enums, and dataclasses used
throughout the OCR application for representing OCR data, settings, and
various configuration options.
"""

from uuid import UUID
from enum import Enum
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from PySide6.QtGui import QImage

class OpStatus(Enum):
    """Operation status indicators for various OCR operations."""
    SUCCESS = 0
    FAILED = 1


class Platform(Enum):
    """Supported operating system platforms."""
    Windows = 0
    Mac = 1
    Linux = 2

class Encoding(Enum):
    """Text encoding formats for OCR output."""
    Unicode = 0
    Wylie = 1

class CharsetEncoder(Enum):
    """Character set encoding methods for OCR models."""
    Wylie = 0
    Stack = 1

class ExportFormat(Enum):
    """Available export formats for OCR results."""
    Text = 0
    XML = 1
    JSON = 2

class Theme(Enum):
    """Application UI theme options."""
    Dark = 0
    Light = 1

class LineMode(Enum):
    """Line detection modes for OCR processing."""
    Line = 0
    Layout = 1

class LineMerge(Enum):
    """Methods for merging detected lines."""
    Merge = 0
    Stack = 1

class LineSorting(Enum):
    """Algorithms for sorting detected lines."""
    Threshold = 0
    Peaks = 1

class OCRArchitecture(Enum):
    """Supported OCR model architectures."""
    Easter2 = 0
    CRNN = 1

class TPSMode(Enum):
    """Thin Plate Spline transformation modes for dewarping."""
    GLOBAL = 0
    LOCAL = 1


class Language(Enum):
    """Supported application languages."""
    English = 0
    German = 1
    French = 2
    Tibetan = 3
    Chinese = 4


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
    NewButton: str
    ImportButton: str
    SaveButton: str
    RunButton: str
    SettingsButton: str

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
