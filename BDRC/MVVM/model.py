import os
import json

import pyewts

from glob import glob
from BDRC.Utils import create_dir, import_local_models, find_key
from BDRC.Data import *
from Config import *


class SettingsModel:

    def __init__(self, user_directory: str, execution_directory: str):
        self.user_directory = user_directory
        self.execution_directory = execution_directory
        self.app_settings, self.ocr_settings, self.export_settings = self.read_settings(self.user_directory)
        self.ocr_models = []
        self.DEFAULT_FONT = os.path.join(self.execution_directory, "Assets", "Fonts", "TibMachUni-1.901b.ttf")

        self.default_models_path = os.path.join(self.user_directory, "Models")
        self.bundled_models_path = os.path.join(self.execution_directory, "OCRModels")
        self.photi_line_model_path = os.path.join(self.execution_directory, "Models", "Lines")
        self.photi_layout_model_path = os.path.join(self.execution_directory, "Models", "Layout")

        self.line_model_config = self.read_line_model_config(self.photi_line_model_path)
        self.layout_model_config = self.read_layout_model_config(self.photi_layout_model_path)

        self.tmp_dir = os.path.join(self.user_directory, "tmp")
        create_dir(self.tmp_dir)

        # First try to load models from user-specified path
        models_loaded = False
        if os.path.isdir(self.app_settings.model_path):
            try:
                ocr_models = import_local_models(self.app_settings.model_path)
                if ocr_models and len(ocr_models) > 0:
                    self.ocr_models = ocr_models
                    models_loaded = True
                    print(f"Loaded {len(ocr_models)} models from user-specified path: {self.app_settings.model_path}")
            except BaseException as e:
                print(f"Error loading models from user path: {str(e)}")
                pass
        
        # If no models were loaded from user path, try the bundled models
        if not models_loaded and os.path.isdir(self.bundled_models_path):
            try:
                ocr_models = import_local_models(self.bundled_models_path)
                if ocr_models and len(ocr_models) > 0:
                    self.ocr_models = ocr_models
                    print(f"Loaded {len(ocr_models)} models from bundled path: {self.bundled_models_path}")
            except BaseException as e:
                print(f"Error loading bundled models: {str(e)}")
                pass

    def clear_temp_files(self):
         # just deleting all tmp files on startup

        if os.path.isdir(self.tmp_dir):
            tmp_files = glob(os.path.join(self.tmp_dir, "*"))

            if len(tmp_files) > 0:
                for file in tmp_files:
                    os.remove(file)
    
    def get_line_model(self):
        if self.ocr_settings.line_mode == LineMode.Line:
            return self.line_model_config
        else:
            return self.layout_model_config

    def update_ocr_settings(self, settings: OCRSettings):
        self.ocr_settings = settings

    def update_app_settings(self, settings: AppSettings):
        self.app_settings = settings

    def update_export_settings(self, settings: ExportSettings):
        self.export_settings = settings

    def create_default_app_config(self, user_dir: str):
        settings = {
                "model_path": os.path.join(self.user_directory, "Models"),
                "language": "en",
                "encoding": "unicode",
                "theme": "dark"
            }
        app_settings_file = os.path.join(user_dir, "app_settings.json")
        with open(app_settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=1)

    def create_default_ocr_config(self, user_dir: str):
        settings = {
            "line_mode": "line",
            "line_merge": "merge",
            "line_sorting": "threshold",
            "dewarp": "yes",
            "merge_lines": "yes",
            "k_factor": 2.5,
            "bbox_tolerance": 3.0,
            "tps": "global",
            "output_encoding": "unicode"
        }

        ocr_settings_file = os.path.join(user_dir, "ocr_settings.json")
        with open(ocr_settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=1)

    def create_default_export_config(self):
        default_settings = ExportSettings(
            ExportFileMode.FilePerPage,
            ExportFormat.Text,
            Encoding.Unicode,
            output_dir="/",
            output_file="",
            before_page_text="",
            # Literal \n\n, not newlines yet
            after_page_text="""\\n\\n"""
        )
        self.save_export_settings(default_settings)

    def read_settings(self, user_dir: str) -> tuple[AppSettings, OCRSettings, ExportSettings]:
        app_settings_file = os.path.join(user_dir, "app_settings.json")
        ocr_settings_file = os.path.join(user_dir, "ocr_settings.json")
        export_settings_file = os.path.join(user_dir, "export_settings.json")

        if not os.path.isfile(app_settings_file):
            self.create_default_app_config(user_dir)

        if not os.path.isfile(ocr_settings_file):
            self.create_default_ocr_config(user_dir)

        if not os.path.isfile(export_settings_file):
            self.create_default_export_config()

        with open(app_settings_file, encoding="utf-8") as file:
            app_json_settings = json.loads(file.read())

        _model_path = app_json_settings["model_path"]
        _lang_code = app_json_settings["language"]
        _encoding = app_json_settings["encoding"]
        _theme = app_json_settings["theme"]

        app_settings = AppSettings(
            model_path=_model_path,
            language=LANGUAGES[_lang_code],
            encoding=ENCODINGS[_encoding],
            theme=THEMES[_theme]
        )

        with open(ocr_settings_file, encoding="utf-8") as file:
            ocr_json_settings = json.loads(file.read())

        _line_mode = ocr_json_settings["line_mode"]
        _line_merge = ocr_json_settings["line_merge"]
        _line_sorting = ocr_json_settings["line_sorting"]
        _k_factor = ocr_json_settings["k_factor"]
        _bbox_tolerance = ocr_json_settings["bbox_tolerance"]
        _dewarping = ocr_json_settings["dewarp"]
        _merge_lines = ocr_json_settings["merge_lines"]
        _tps = ocr_json_settings["tps"]
        _out_encoding = ocr_json_settings["output_encoding"]

        ocr_settings = OCRSettings(
            line_mode=LINE_MODES[_line_mode],
            line_merge=LINE_MERGE[_line_merge],
            line_sorting=LINE_SORTING[_line_sorting],
            dewarping=_dewarping == "yes",
            merge_lines=_merge_lines == "yes",
            k_factor=float(_k_factor),
            bbox_tolerance=float(_bbox_tolerance),
            tps_mode=TPS_MODE[_tps],
            output_encoding=ENCODINGS[_out_encoding],
        )

        with open(export_settings_file, encoding="utf-8") as file:
            export_json_settings = json.loads(file.read())

        _export_file_mode = EXPORT_FILE_MODES[export_json_settings['file_mode']]
        _export_encoding = ENCODINGS[export_json_settings['encoding']]
        _export_format = EXPORTERS[export_json_settings['format']]

        export_settings = ExportSettings(
            file_mode=_export_file_mode,
            format=_export_format,
            encoding=_export_encoding,
            output_dir=export_json_settings['output_dir'],
            output_file=export_json_settings['output_file'],
            before_page_text=export_json_settings['before_page_text'],
            after_page_text=export_json_settings['after_page_text']
        )

        return app_settings, ocr_settings, export_settings
  
    def save_app_settings(self, settings: AppSettings):
        _model_path = settings.model_path
        _language = [x for x in LANGUAGES if LANGUAGES[x] == settings.language][0]
        _encoding = [x for x in ENCODINGS if ENCODINGS[x] == settings.encoding][0]
        _theme = [x for x in THEMES if THEMES[x] == settings.theme][0]

        _settings = {
                    "model_path": _model_path,
                    "language": _language,
                    "encoding": _encoding,
                    "theme": _theme
                }

        app_settings_file = os.path.join(self.user_directory, "app_settings.json")
        with open(app_settings_file, "w", encoding="utf-8") as f:
            json.dump(_settings, f, ensure_ascii=False, indent=1)


    def save_ocr_settings(self, settings: OCRSettings):
        _line_mode = [x for x in LINE_MODES if LINE_MODES[x] == settings.line_mode][0]
        _line_merge = [x for x in LINE_MERGE if LINE_MERGE[x] == settings.line_merge][0]
        _line_sorting = [x for x in LINE_SORTING if LINE_SORTING[x] == settings.line_sorting][0]
        _dewarp = "yes" if settings.dewarping else "no"
        _merge_lines = "yes" if settings.merge_lines else "no"
        _tps = [x for x in TPS_MODE if TPS_MODE[x] == settings.tps_mode][0]

        _settings = {
            "line_mode": _line_mode,
            "line_merge": _line_merge,
            "line_sorting": _line_sorting,
            "dewarp": _dewarp,
            "merge_lines": _merge_lines,
            "k_factor": settings.k_factor,
            "bbox_tolerance": settings.bbox_tolerance,
            "tps": _tps,
            "output_encoding": "unicode",
        }

        ocr_settings_file = os.path.join(self.user_directory, "ocr_settings.json")
        with open(ocr_settings_file, "w", encoding="utf-8") as f:
            json.dump(_settings, f, ensure_ascii=False, indent=1)

    def save_export_settings(self, settings: ExportSettings):
        _format = find_key(settings.format, EXPORTERS)
        _encoding = find_key(settings.encoding, ENCODINGS)

        settings = {
            "file_mode": settings.file_mode.name,
            "format": _format,
            "encoding": _encoding,
            "output_dir": settings.output_dir,
            "output_file": settings.output_file,
            "before_page_text": settings.before_page_text,
            "after_page_text": settings.after_page_text
        }

        export_settings_file = os.path.join(self.user_directory, "export_settings.json")
        with open(export_settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=1)

    def read_line_model_config(self, target_dir: str) -> LineDetectionConfig:
        target_file = os.path.join(target_dir, "config.json")
        model_dir = os.path.dirname(target_file)
        file = open(target_file, encoding="utf-8")
        json_content = json.loads(file.read())

        onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
        patch_size = int(json_content["patch_size"])

        config = LineDetectionConfig(onnx_model_file, patch_size)

        return config
    
    def read_layout_model_config(self, target_dir: str) -> LayoutDetectionConfig:
        target_file = os.path.join(target_dir, "config.json")
        model_dir = os.path.dirname(target_file)
        file = open(target_file, encoding="utf-8")
        json_content = json.loads(file.read())

        onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
        patch_size = int(json_content["patch_size"])
        classes = json_content["classes"]

        config = LayoutDetectionConfig(onnx_model_file, patch_size, classes)

        return config


class OCRDataModel:
    def __init__(self):
        self.data = {}
        self.converter = pyewts.pyewts()

    def add_data(self, data: Dict[UUID, OCRData]):
        self.data.clear()
        self.data = data

    def get_data(self):
        data = list(self.data.values())
        return data

    def clear_data(self):
        self.data.clear()

    def add_page_data(
        self, guid: UUID, lines: List[Line], preview_image: npt.NDArray, angle: float
    ) -> None:
        self.data[guid].lines = lines
        self.data[guid].preview = preview_image
        self.data[guid].angle = angle

    def add_ocr_text(self, guid: UUID, ocr_lines: List[OCRLine]):
        self.data[guid].ocr_lines = ocr_lines

    def delete_image(self, guid: UUID):
        del self.data[guid]

    def convert_wylie_unicode(self, guid: UUID):
        for ocr_line in self.data[guid].ocr_lines:
            if ocr_line.encoding == Encoding.Wylie:
                new_text = self.converter.toUnicode(ocr_line.text)
                ocr_line.text = new_text
                ocr_line.encoding = Encoding.Unicode
            else:
                new_text = self.converter.toWylie(ocr_line.text)
                ocr_line.text = new_text
                ocr_line.encoding = Encoding.Wylie

    def update_ocr_line(self, ocr_line_update: OCRLineUpdate):
        for ocr_line in self.data[ocr_line_update.page_guid].lines:
            if ocr_line.guid == ocr_line_update.ocr_line.guid:
                ocr_line.text = ocr_line_update.ocr_line.text
                ocr_line.encoding = ocr_line_update.ocr_line.encoding
