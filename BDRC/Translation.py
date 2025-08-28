"""
Translation manager for the Tibetan OCR App.
Handles internationalization using Qt's translation system.
"""

import os
from typing import Optional
from PySide6.QtCore import QTranslator, QLocale, QCoreApplication


class TranslationManager:
    """Manages application translations and language switching."""
    
    def __init__(self, app, base_dir: str):
        self.app = app
        self.base_dir = base_dir
        self.translations_dir = os.path.join(base_dir, "translations")
        self.current_translator: Optional[QTranslator] = None
        self.current_language = "en"  # Default to English
        
        # Create translations directory if it doesn't exist
        os.makedirs(self.translations_dir, exist_ok=True)
    
    def get_available_languages(self) -> dict:
        """Get available language codes and their display names."""
        return {
            "en": "English",
            "bo": "བོད་ཡིག" # Tibetan script for "Tibetan language"
        }
    
    def load_translation(self, language_code: str) -> bool:
        """Load translation for the specified language code."""
        if language_code == "en":
            # English is the base language, remove any existing translator
            if self.current_translator:
                self.app.removeTranslator(self.current_translator)
                self.current_translator = None
            self.current_language = "en"
            return True
        
        # Remove existing translator if any
        if self.current_translator:
            self.app.removeTranslator(self.current_translator)
        
        # Create new translator
        translator = QTranslator()
        
        # Try to load translation file
        translation_file = os.path.join(self.translations_dir, f"{language_code}.qm")
        
        if os.path.exists(translation_file):
            success = translator.load(translation_file)
        else:
            # Try loading from Qt locale format
            success = translator.load(QLocale(language_code), "tibetan_ocr", "_", self.translations_dir)
        
        if success:
            self.app.installTranslator(translator)
            self.current_translator = translator
            self.current_language = language_code
            return True
        else:
            print(f"Failed to load translation for language: {language_code}")
            return False
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        return self.current_language
    
    def switch_language(self, language_code: str) -> bool:
        """Switch to a different language and update the UI."""
        success = self.load_translation(language_code)
        if success:
            # Force UI update by emitting language change event
            # This will trigger retranslateUi methods in widgets that implement it
            QCoreApplication.instance().processEvents()
        return success


def tr(text: str) -> str:
    """
    Translation function wrapper for easier use throughout the application.
    
    Args:
        text: The text to translate
        
    Returns:
        Translated text if available, otherwise the original text
    """
    return QCoreApplication.translate("", text)