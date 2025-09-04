
"""
Main entry point for the Tibetan OCR Desktop Application.

This script initializes and runs the OCR application using the MVVM pattern
with PySide6 for the GUI framework. It sets up the application models,
view models, and main view, handles translation management, and manages
application settings.

The application supports:
- Multiple OCR models for different Tibetan scripts
- Line and layout detection modes
- Text encoding conversion (Wylie/Unicode)
- Multiple export formats (Text, XML, JSON)
- Internationalization support

Build commands for different platforms are included at the top of the file.
"""

import os
import sys
from platformdirs import user_data_dir
from PySide6.QtCore import QPoint
from BDRC.MVVM.view import AppView
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from BDRC.Utils import get_screen_center, get_platform, create_dir
from PySide6.QtWidgets import QApplication
from BDRC.Styles import DARK
from BDRC.Translation import TranslationManager

APP_NAME = "BDRC_OCR"
APP_AUTHOR = "BDRC"


if __name__ == "__main__":
    """
    Application initialization and startup sequence.
    
    1. Detect platform and setup directories
    2. Initialize Qt application with dark theme
    3. Setup translation management
    4. Create and configure data and settings models
    5. Initialize view models
    6. Create and display main application window
    7. Start Qt event loop
    """
    platform = get_platform()
    execution_dir= os.path.dirname(__file__)
    udi = user_data_dir(APP_NAME, APP_AUTHOR)
    create_dir(udi)
    
    app = QApplication()
    app.setStyleSheet(DARK)

    # Initialize translation manager
    translation_manager = TranslationManager(app, execution_dir)
    
    # Load saved language preference or default to English
    from PySide6.QtCore import QSettings
    settings = QSettings("BDRC", "TibetanOCRApp")
    saved_language = settings.value("ui/language", "en")
    translation_manager.load_translation(saved_language)

    data_model = OCRDataModel()
    settings_model = SettingsModel(udi, execution_dir)

    dataview_model = DataViewModel(data_model)
    settingsview_model = SettingsViewModel(settings_model)

    screen_data = get_screen_center(app)

    app_view = AppView(
        dataview_model,
        settingsview_model,
        platform,
        translation_manager
    )

    app_view.resize(screen_data.start_width, screen_data.start_height)
    app_view.move(QPoint(screen_data.start_x, screen_data.start_y))

    sys.exit(app.exec())
