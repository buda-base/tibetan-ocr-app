"""
View model classes for the Tibetan OCR application following MVVM pattern.

This module contains view model classes that serve as intermediaries between
the models and views, handling UI interactions and data binding with Qt signals.
"""

from uuid import UUID
import numpy.typing as npt
from typing import List, Dict
from PySide6.QtCore import QObject, Signal
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.Data import OCRData, Line, OCRLine, OCRLineUpdate, OCRModel, AppSettings, OCRSettings


class SettingsViewModel(QObject):
    """
    View model for application and OCR settings management.
    
    Handles settings-related UI interactions and provides signals for
    notifying the view of configuration changes.
    """
    s_app_settings_changed = Signal(AppSettings)
    s_ocr_settings_changed = Signal(OCRSettings)
    s_ocr_models_changed = Signal()
    s_ocr_model_changed = Signal(OCRModel)

    def __init__(self, model: SettingsModel):
        """
        Initialize the settings view model.
        
        Args:
            model: SettingsModel instance to manage
        """
        super().__init__()
        self._model = model

    def get_tmp_dir(self):
        """Get the temporary directory path."""
        return self._model.tmp_dir

    def get_execution_dir(self) -> str:
        """Get the application execution directory path."""
        return self._model.execution_directory
    
    def get_default_font_path(self) -> str:
        """Get the path to the default font file."""
        return self._model.DEFAULT_FONT
    
    def get_line_model(self):
        """Get the current line detection model configuration."""
        return self._model.get_line_model()
    
    def get_ocr_models(self):
        """Get the list of available OCR models."""
        return self._model.ocr_models

    def get_current_ocr_model(self) -> OCRModel | None:
        """Return the persisted OCRModel or fallback to first."""
        from PySide6.QtCore import QSettings
        if self._model.ocr_models:
            settings = QSettings("BDRC", "TibetanOCRApp")
            name = settings.value("main/model_name", None)
            if name is not None:
                for m in self._model.ocr_models:
                    if m.name == name:
                        return m
            return self._model.ocr_models[0]
        return None

    def get_ocr_settings(self) -> OCRSettings:
        """Get the current OCR processing settings."""
        return self._model.ocr_settings

    def get_app_settings(self) -> AppSettings:
        """Get the current application settings."""
        return self._model.app_settings

    def update_ocr_settings(self, settings: OCRSettings):
        """
        Update OCR settings and emit change signal.
        
        Args:
            settings: New OCR settings to apply
        """
        self._model.update_ocr_settings(settings)
        self.s_ocr_settings_changed.emit(settings)

    def update_app_settings(self, settings: AppSettings):
        """
        Update application settings and emit change signal.
        
        Args:
            settings: New application settings to apply
        """
        self._model.update_app_settings(settings)
        self.s_app_settings_changed.emit(settings)

    def update_ocr_models(self, ocr_models: List[OCRModel]):
        """
        Update the list of available OCR models.
        
        Args:
            ocr_models: New list of OCR models
        """
        if len(ocr_models) > 0:
            self._model.ocr_models = ocr_models
            self.s_ocr_models_changed.emit()

    def select_ocr_model(self, ocr_model: OCRModel):
        """
        Select a specific OCR model and emit selection signal.
        
        Args:
            ocr_model: OCR model to select
        """
        self.s_ocr_model_changed.emit(ocr_model)

    def save_app_settings(self, settings: AppSettings):
        """
        Save application settings to file.
        
        Args:
            settings: Application settings to save
        """
        self._model.save_app_settings(settings)

    def save_ocr_settings(self, settings: OCRSettings):
        """
        Save OCR settings to file.
        
        Args:
            settings: OCR settings to save
        """
        self._model.save_ocr_settings(settings)


class DataViewModel(QObject):
    """
    View model for managing OCR data and image processing results.
    
    Handles OCR data interactions, manages data selection and updates,
    and provides signals for notifying the view of data changes.
    
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
        """
        Initialize the data view model.
        
        Args:
            model: OCRDataModel instance to manage
        """
        super().__init__()
        self._model = model

    def get_data_by_guid(self, guid: UUID) -> OCRData:
        """
        Retrieve OCR data for a specific image by its GUID.
        
        Args:
            guid: Unique identifier for the image
            
        Returns:
            OCRData instance for the specified image
        """
        return self._model.data[guid]

    def get_data(self) -> Dict[UUID, OCRData]:
        """Get all OCR data as a dictionary mapping GUIDs to OCRData."""
        return self._model.data

    def add_data(self, data: Dict[UUID, OCRData]):
        """
        Add new OCR data and emit data change signals.
        
        Args:
            data: Dictionary mapping GUIDs to OCRData instances
        """
        self.clear_data()
        self._model.add_data(data)

        current_data = self._model.get_data()
        self.s_data_changed.emit(current_data)

    def select_data_by_guid(self, uuid: UUID):
        """
        Select OCR data by GUID and emit selection signal.
        
        Args:
            uuid: GUID of the image to select
        """
        self.s_data_selected.emit(self._model.data[uuid])

    def delete_image_by_guid(self, guid: UUID):
        """
        Delete OCR data for a specific image and emit size change signal.
        
        Args:
            guid: GUID of the image to delete
        """
        self._model.delete_image(guid)
        self.s_data_size_changed.emit(self._model.get_data())

    def get_data_index(self, uuid: UUID):
        """
        Get the index position of OCR data by GUID.
        
        Args:
            uuid: GUID to find index for
            
        Returns:
            Index position in the data collection
        """
        _entries = list(self._model.data.keys())
        return _entries.index(uuid)

    def select_data_by_index(self, index: int):
        """
        Select OCR data by index position (used by PageSwitcher).
        
        Args:
            index: Index position to select
        """
        # This is the case when an index is fed by the PageSwitcher
        current_data = list(self._model.data.values())
        self.s_data_auto_selected.emit(current_data[index])

    def update_ocr_data(self, uuid: UUID, ocr_lines: List[OCRLine], silent: bool = False):
        """
        Update OCR text results for a specific image.
        
        Args:
            uuid: GUID of the image to update
            ocr_lines: New OCR text lines
            silent: If True, don't emit change signals
        """
        self._model.add_ocr_text(uuid, ocr_lines)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.s_record_changed.emit(data)

    def update_page_data(self, uuid: UUID, lines: List[Line], preview_image: npt.NDArray, angle: float, silent: bool = False):
        """
        Update line detection results for a specific image.
        
        Args:
            uuid: GUID of the image to update
            lines: Detected text lines
            preview_image: Preview image with line overlays
            angle: Detected rotation angle
            silent: If True, don't emit change signals
        """
        self._model.add_page_data(uuid, lines, preview_image, angle)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.s_page_data_update.emit(data)

    def update_ocr_line(self, ocr_line_update: OCRLineUpdate):
        """
        Update a specific OCR text line and emit update signal.
        
        Args:
            ocr_line_update: Update containing page GUID and modified OCR line
        """
        self._model.update_ocr_line(ocr_line_update)
        self.s_ocr_line_update.emit(self._model.data[ocr_line_update.page_guid])

    def convert_wylie_unicode(self, page_guid: UUID):
        """
        Convert text encoding between Wylie and Unicode for a page.
        
        Args:
            page_guid: GUID of the page to convert
        """
        self._model.convert_wylie_unicode(page_guid)

        data = self.get_data_by_guid(page_guid)
        self.s_record_changed.emit(data)

    def clear_data(self):
        """Clear all OCR data and emit cleared signal."""
        self._model.clear_data()
        self.s_data_cleared.emit()
