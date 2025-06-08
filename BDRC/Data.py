from uuid import UUID
from enum import Enum
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple, NamedTuple
from PySide6.QtGui import QImage

class OpStatus(Enum):
    SUCCESS = 0
    FAILED = 1


class Platform(Enum):
    Windows = 0
    Mac = 1
    Linux = 2

class Encoding(Enum):
    Unicode = 0
    Wylie = 1

class CharsetEncoder(Enum):
    Wylie = 0
    Stack = 1

class ExportFormat(Enum):
    Text = 0
    PageXML = 1
    JSONLines = 2

class ExportFileMode(Enum):
    FilePerPage = (0, "File per page")
    OneBigFile = (1, "One big file")

    label: str

    def __new__(cls, ord: int, label: str):
        value = object.__new__(cls)
        value._value_ = ord
        value.label = label
        return value

class Theme(Enum):
    Dark = 0
    Light = 1


class LineMode(Enum):
    Line = 0
    Layout = 1


class LineMerge(Enum):
    Merge = 0
    Stack = 1


class LineSorting(Enum):
    Threshold = 0
    Peaks = 1

class OCRArchitecture(Enum):
    Easter2 = 0
    CRNN = 1

class TPSMode(Enum):
    GLOBAL = 0
    LOCAL = 1


class Language(Enum):
    English = 0
    German = 1
    French = 2
    Tibetan = 3
    Chinese = 4


@dataclass
class ScreenData:
    max_width: int
    max_height: int
    start_width: int
    start_height: int
    start_x: int
    start_y: int

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Line:
    guid: UUID
    contour: npt.NDArray
    bbox: BBox
    center: Tuple[int, int]

@dataclass
class OCRLine:
    guid: UUID
    text: str
    encoding: Encoding

@dataclass
class TransformedContours:
    plain_box: str
    plain_lines: list[tuple[int, int, int, int] | str]

@dataclass
class LayoutData:
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
    name: str
    NewButton: str
    ImportButton: str
    SaveButton: str
    RunButton: str
    SettingsButton: str

@dataclass
class OCRData:
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
    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    model_file: str
    patch_size: int
    classes: List[str]


@dataclass
class OCRModelConfig:
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
    guid: UUID
    lines: List[Line]


@dataclass
class OCRLineUpdate:
    page_guid: UUID
    ocr_line: OCRLine

@dataclass
class OCRLineEncodingUpdate:
    page_guid: UUID
    ocr_lines: List[OCRLine]

@dataclass
class OCResult:
    guid: UUID
    mask: npt.NDArray
    lines: List[Line]
    text: List[OCRLine]
    angle: float

@dataclass
class OCRSample:
    cnt: int
    guid: UUID
    name: str
    result: OCResult


@dataclass
class OCRModel:
    guid: UUID
    name: str
    path: str
    config: OCRModelConfig

@dataclass
class OCRSettings:
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
    model_path: str
    language: Language
    encoding: Encoding
    theme: Theme

@dataclass
class ExportSettings:
    file_mode: ExportFileMode
    format: ExportFormat
    encoding: Encoding
    output_dir: str
    output_file: str
    before_page_text: str
    after_page_text: str
