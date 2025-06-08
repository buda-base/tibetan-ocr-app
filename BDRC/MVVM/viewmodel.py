from uuid import UUID
import numpy.typing as npt
from typing import List, Dict
from PySide6.QtCore import QObject, Signal
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.Data import OCRData, Line, OCRLine, OCRLineUpdate, OCRModel, AppSettings, OCRSettings, ExportSettings


class SettingsViewModel(QObject):
    s_app_settings_changed = Signal(AppSettings)
    s_ocr_settings_changed = Signal(OCRSettings)
    s_export_settings_changed = Signal(ExportSettings)

    s_ocr_models_changed = Signal()
    s_ocr_model_changed = Signal(OCRModel)

    def __init__(self, model: SettingsModel):
        super().__init__()
        self._model = model

    def get_tmp_dir(self):
        return self._model.tmp_dir

    def get_execution_dir(self) -> str:
        return self._model.execution_directory
    
    def get_default_font_path(self) -> str:
        return self._model.DEFAULT_FONT
    
    def get_line_model(self):
        return self._model.get_line_model()
    
    def get_ocr_models(self):
        return self._model.ocr_models

    def get_current_ocr_model(self) -> OCRModel | None:
        if len(self._model.ocr_models) > 0:
            return self._model.ocr_models[0]
        else:
            return None

    def get_ocr_settings(self) -> OCRSettings:
        return self._model.ocr_settings

    def get_app_settings(self) -> AppSettings:
        return self._model.app_settings

    def get_export_settings(self) -> ExportSettings:
        return self._model.export_settings

    def update_ocr_settings(self, settings: OCRSettings):
        self._model.update_ocr_settings(settings)
        self.s_ocr_settings_changed.emit(settings)

    def update_app_settings(self, settings: AppSettings):
        self._model.update_app_settings(settings)
        self.s_app_settings_changed.emit(settings)

    def update_export_settings(self, settings: ExportSettings):
        self._model.update_export_settings(settings)
        self.s_export_settings_changed.emit(settings)

    def update_ocr_models(self, ocr_models: List[OCRModel]):
        if len(ocr_models) > 0:
            self._model.ocr_models = ocr_models
            self.s_ocr_models_changed.emit()

    def select_ocr_model(self, ocr_model: OCRModel):
        self.s_ocr_model_changed.emit(ocr_model)

    def save_app_settings(self, settings: AppSettings):
        self._model.save_app_settings(settings)

    def save_ocr_settings(self, settings: OCRSettings):
        self._model.save_ocr_settings(settings)

    def save_export_settings(self, settings: ExportSettings):
        self._model.save_export_settings(settings)


class DataViewModel(QObject):
    """
   Note: The dataAutoSelected Signal is a temporary workaround to handle the case of a data record being selected
   via the page switcher in the header, which focuses and scrolls to the respective image in the ImageGallery.
   This is for time being a separate signal to avoid having a cycling signal when an image gets selected in the ImageGallery
   via seleced_by_guid which would be focused afterwards as well - which is a weird behaviour
    """

    s_record_changed = Signal(OCRData)
    s_page_data_update = Signal(OCRData)
    s_data_selected = Signal(OCRData)
    s_data_changed = Signal(list)
    s_data_size_changed = Signal(list)
    s_ocr_line_update = Signal(OCRData) # for TextView

    s_data_auto_selected = Signal(OCRData)
    s_data_cleared = Signal()

    def __init__(self, model: OCRDataModel):
        super().__init__()
        self._model = model

    def get_data_by_guid(self, guid: UUID) -> OCRData:
        return self._model.data[guid]

    def get_data(self) -> Dict[UUID, OCRData]:
        return self._model.data

    def add_data(self, data: Dict[UUID, OCRData]):
        self.clear_data()
        self._model.add_data(data)

        current_data = self._model.get_data()
        self.s_data_changed.emit(current_data)

    def select_data_by_guid(self, uuid: UUID):
        self.s_data_selected.emit(self._model.data[uuid])

    def delete_image_by_guid(self, guid: UUID):
        self._model.delete_image(guid)
        self.s_data_size_changed.emit(self._model.get_data())

    def get_data_index(self, uuid: UUID):
        _entries = list(self._model.data.keys())
        return _entries.index(uuid)

    def select_data_by_index(self, index: int):
        # This is the case when an index is fed by the PageSwitcher
        current_data = list(self._model.data.values())
        self.s_data_auto_selected.emit(current_data[index])

    def update_ocr_data(self, uuid: UUID, ocr_lines: List[OCRLine], silent: bool = False):
        self._model.add_ocr_text(uuid, ocr_lines)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.s_record_changed.emit(data)

    def update_page_data(self, uuid: UUID, lines: List[Line], preview_image: npt.NDArray, angle: float, silent: bool = False):
        self._model.add_page_data(uuid, lines, preview_image, angle)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.s_page_data_update.emit(data)

    def update_ocr_line(self, ocr_line_update: OCRLineUpdate):
        self._model.update_ocr_line(ocr_line_update)
        self.s_ocr_line_update.emit(self._model.data[ocr_line_update.page_guid])

    def convert_wylie_unicode(self, page_guid: UUID):
        self._model.convert_wylie_unicode(page_guid)

        data = self.get_data_by_guid(page_guid)
        self.s_record_changed.emit(data)

    def clear_data(self):
        self._model.clear_data()
        self.s_data_cleared.emit()
