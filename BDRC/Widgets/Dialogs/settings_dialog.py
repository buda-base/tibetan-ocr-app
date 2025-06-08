import os
from typing import List, Tuple
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QButtonGroup,
    QRadioButton,
    QTabWidget,
    QLineEdit,
    QFileDialog,
    QMessageBox
)
from PySide6.QtGui import QColor

from BDRC.Data import AppSettings, OCRSettings, OCRModel, Theme, LineMode, Language, Encoding
from BDRC.Utils import import_local_models
from BDRC.Widgets.Dialogs.helpers import (
    build_line_mode,
    build_encodings,
    build_languages,
    build_binary_selection
)

class SettingsDialog(QDialog):
    def __init__(
            self,
            app_settings: AppSettings,
            ocr_settings: OCRSettings,
            ocr_models: List[OCRModel],
        ):
        super().__init__()
        self.setObjectName("SettingsDialog")
        self.app_settings = app_settings
        self.ocr_settings = ocr_settings
        self.ocr_models = ocr_models
        self.selected_theme = Theme.Dark
        self.selected_exporters = []

        # Settings
        # Theme
        self.dark_theme_btn = QRadioButton("Dark")
        self.light_theme_btn = QRadioButton("Light")

        self.dark_theme_btn.setObjectName("OptionsRadio")
        self.light_theme_btn.setObjectName("OptionsRadio")

        self.theme_group = QButtonGroup()
        self.theme_group.setObjectName("OptionsRadio")
        self.theme_group.setExclusive(True)
        self.theme_group.addButton(self.dark_theme_btn)
        self.theme_group.addButton(self.light_theme_btn)
        self.theme_group.setId(self.dark_theme_btn, Theme.Dark.value)
        self.theme_group.setId(self.light_theme_btn, Theme.Light.value)

        if self.app_settings.theme == Theme.Dark:
            self.dark_theme_btn.setChecked(True)
            self.light_theme_btn.setChecked(False)
        else:
            self.dark_theme_btn.setChecked(False)
            self.light_theme_btn.setChecked(True)

        self.import_models_btn = QPushButton("Import Models")
        self.import_models_btn.setObjectName("SmallDialogButton")
        self.import_models_btn.clicked.connect(self.handle_model_import)

        self.line_mode_group, self.line_mode_buttons = build_line_mode(
            self.ocr_settings.line_mode
        )

        self.encodings_group, self.encoding_buttons = build_encodings(
            self.app_settings.encoding
        )
        self.language_group, self.language_buttons = build_languages(
            self.app_settings.language
        )
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(
            self.ocr_settings.dewarping
        )
        self.merge_group, self.merge_buttons = build_binary_selection(
            self.ocr_settings.merge_lines
        )

        self.setWindowTitle("BDRC Settings")
        self.setMinimumHeight(460)
        self.setMinimumWidth(800)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # define layout
        self.settings_tabs = QTabWidget()

        # General Settings Tab
        self.general_settings_tab = QWidget()

        theme_label = QLabel("UI Theme")
        theme_label.setFixedWidth(100)
        theme_label.setObjectName("OptionsLabel")

        ui_theme_layout = QHBoxLayout()
        ui_theme_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ui_theme_layout.addWidget(theme_label)
        ui_theme_layout.addWidget(self.dark_theme_btn)
        ui_theme_layout.addWidget(self.light_theme_btn)

        language_layout = QHBoxLayout()
        language_label = QLabel("Language")
        language_label.setObjectName("OptionsLabel")
        language_label.setFixedWidth(100)
        language_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        language_layout.addWidget(language_label)

        for btn in self.language_buttons:
            language_layout.addWidget(btn)

        self.general_settings_layout = QVBoxLayout()
        self.general_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.general_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.general_settings_layout.setSpacing(20)

        self.general_settings_layout.addLayout(ui_theme_layout)
        self.general_settings_layout.addLayout(language_layout)
        self.general_settings_tab.setLayout(self.general_settings_layout)

        # OCR Models Tab
        self.ocr_models_tab = QWidget()

        self.models_data_table = QTableWidget()
        self.models_data_table.setObjectName("ModelTable")
        self.models_data_table.setColumnCount(5)
        self.models_data_table_header = [
            "Model",
            "Encoding",
            "Architecture",
            "Version",
            "Model file",
        ]
        self.models_data_table.setAutoScroll(True)
        self.models_data_table.horizontalHeader().setStretchLastSection(True)

        models_data_table_label = QLabel("Available OCR Models")
        models_data_table_label.setObjectName("OptionsLabel")

        h_layout = QHBoxLayout()
        h_layout.addWidget(models_data_table_label)
        h_layout.addWidget(self.import_models_btn)

        ocr_models_tab_layout = QVBoxLayout()
        ocr_models_tab_layout.addLayout(h_layout)
        ocr_models_tab_layout.addWidget(self.models_data_table)
        self.ocr_models_tab.setLayout(ocr_models_tab_layout)

        # OCR Settings Tab
        self.ocr_settings_tab = QWidget()
        self.ocr_settings_tab.setContentsMargins(0, 20, 0, 0)
        self.ocr_settings_layout = QVBoxLayout()
        self.ocr_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.ocr_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # line model
        line_mode_layout = QHBoxLayout()
        line_mode_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        line_mode_label = QLabel("Line Model")
        line_mode_label.setObjectName("OptionsLabel")
        line_mode_layout.addWidget(line_mode_label)

        for line_mode_btn in self.line_mode_buttons:
            line_mode_layout.addWidget(line_mode_btn)

        # encoding
        encoding_layout = QHBoxLayout()
        encoding_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        encoding_label = QLabel("Encoding")
        encoding_label.setObjectName("OptionsLabel")
        encoding_layout.addWidget(encoding_label)

        for encoding in self.encoding_buttons:
            encoding_layout.addWidget(encoding)

        # dewarping
        dewarping_layout = QHBoxLayout()
        dewarping_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        dewarping_label = QLabel("Dewarping")
        dewarping_label.setObjectName("OptionsLabel")
        dewarping_layout.addWidget(dewarping_label)
        for btn in self.dewarp_buttons:
            dewarping_layout.addWidget(btn)

        # merge ocr_lines
        merge_layout = QHBoxLayout()
        merge_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        merge_label = QLabel("Merge")
        merge_label.setObjectName("OptionsLabel")
        merge_layout.addWidget(merge_label)

        for btn in self.merge_buttons:
            merge_layout.addWidget(btn)

        # specific ocr parameters
        k_factor_layout = QHBoxLayout()
        k_factor_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        k_factor_label = QLabel("K-Factor")
        k_factor_label.setFixedWidth(160)
        k_factor_label.setObjectName("OptionsLabel")

        self.k_factor_edit = QLineEdit()
        self.k_factor_edit.setFixedWidth(60)
        self.k_factor_edit.setObjectName("DialogLineEdit")
        self.k_factor_edit.setText(str(self.ocr_settings.k_factor))
        self.k_factor_edit.editingFinished.connect(self.validate_kfactor_input)
        k_factor_layout.addWidget(k_factor_label)
        k_factor_layout.addWidget(self.k_factor_edit)

        bbox_tolerance_layout = QHBoxLayout()
        bbox_tolerance_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        bbox_tolerance_label = QLabel("Bounding Box Tolerance")
        bbox_tolerance_label.setFixedWidth(160)
        bbox_tolerance_label.setObjectName("OptionsLabel")
        self.bbox_tolerance_edit = QLineEdit()
        self.bbox_tolerance_edit.setObjectName("DialogLineEdit")
        self.bbox_tolerance_edit.setFixedWidth(60)
        self.bbox_tolerance_edit.editingFinished.connect(
            self.validate_bbox_tolerance_input
        )
        self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        bbox_tolerance_layout.addWidget(bbox_tolerance_label)
        bbox_tolerance_layout.addWidget(self.bbox_tolerance_edit)

        line_mode_label.setFixedWidth(160)
        encoding_label.setFixedWidth(160)
        dewarping_label.setFixedWidth(160)
        merge_label.setFixedWidth(160)

        explanation = QLabel(
            """
            The above settings give some control over the OCR process. The <b>k Factor</b> is a parameter to control the intensity of the
            line extraction if adjustments are needed. If you get poor OCR results, try to increase or decrease the value.
            Similarly, the <b>bbox tolerance</b> parameter is the amount of the 'initial line detection' (highlighted in orange after OCR)
            is admissive in order to interpret the rest of the line such as descenders or vowels. A high value can cause problems on pages with a tight
            layout."""
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("OptionsExplanation")

        # assemlbe all layouts
        self.ocr_settings_layout.addLayout(line_mode_layout)
        self.ocr_settings_layout.addLayout(encoding_layout)
        self.ocr_settings_layout.addLayout(dewarping_layout)
        self.ocr_settings_layout.addLayout(merge_layout)
        self.ocr_settings_layout.addLayout(k_factor_layout)
        self.ocr_settings_layout.addLayout(bbox_tolerance_layout)
        self.ocr_settings_layout.addWidget(explanation)
        self.ocr_settings_tab.setLayout(self.ocr_settings_layout)

        # build entire Layout
        # self.settings_tabs.addTab(self.general_settings_tab, "General")
        self.settings_tabs.addTab(self.ocr_models_tab, "OCR Models")
        self.settings_tabs.addTab(self.ocr_settings_tab, "OCR Settings")

        self.main_v_layout = QVBoxLayout()
        self.main_v_layout.addWidget(self.settings_tabs)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)
        self.main_v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.main_v_layout)

        # bind signals
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.update_model_table(self.ocr_models)

    def update_model_table(self, ocr_models: List[OCRModel]):
        self.models_data_table.clear()
        self.models_data_table.setRowCount(len(ocr_models))
        self.models_data_table.setHorizontalHeaderLabels(self.models_data_table_header)

        for i, model in enumerate(ocr_models):
            self.add_ocr_model(i, model)

    def add_ocr_model(self, row_idx: int, ocr_model: OCRModel):
        model_name = QTableWidgetItem(ocr_model.name)
        model_name.setForeground(QColor("#ffffff"))
        self.models_data_table.setItem(row_idx, 0, model_name)

        model_encoding = QTableWidgetItem(ocr_model.config.encoder.name)
        model_encoding.setForeground(QColor("#ffffff"))
        self.models_data_table.setItem(row_idx, 1, model_encoding)

        model_arch = QTableWidgetItem(ocr_model.config.architecture.name)
        model_arch.setForeground(QColor("#ffffff"))
        self.models_data_table.setItem(row_idx, 2, model_arch)

        model_version = QTableWidgetItem(str(ocr_model.config.version))
        model_version.setForeground(QColor("#ffffff"))
        self.models_data_table.setItem(row_idx, 3, model_version)

        model_file = QTableWidgetItem(ocr_model.path)
        model_file.setForeground(QColor("#ffffff"))
        self.models_data_table.setItem(row_idx, 4, model_file)

    def validate_bbox_tolerance_input(self):
        try:
            value = float(self.bbox_tolerance_edit.text())
            if value < 0.0 or value > 1.0:
                self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        except ValueError:
            self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))

    def validate_kfactor_input(self):
        try:
            value = float(self.k_factor_edit.text())
            if value < 0.0 or value > 1.0:
                self.k_factor_edit.setText(str(self.ocr_settings.k_factor))
        except ValueError:
            self.k_factor_edit.setText(str(self.ocr_settings.k_factor))

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()

    def clear_models(self):
        self.models_data_table.clear()
        self.models_data_table.setRowCount(0)

    def handle_model_import(self):
        # Create a file dialog to select a directory
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_dialog.setWindowTitle("Select Model Directory")
        
        if file_dialog.exec():
            selected_dir = file_dialog.selectedFiles()[0]
            
            if os.path.isdir(selected_dir):
                try:
                    # Import models from the selected directory
                    imported_models = import_local_models(selected_dir)
                    
                    # Confirm with the user
                    confirm_dialog = QMessageBox()
                    confirm_dialog.setWindowTitle("Confirm Model Import")
                    confirm_dialog.setText("Do you want to import the selected models? Existing models will be replaced.")
                    confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    confirm_dialog.setDefaultButton(QMessageBox.No)
                    
                    if confirm_dialog.exec() == QMessageBox.Yes:
                        self.ocr_models = imported_models
                        self.update_model_table(self.ocr_models)
                        
                except Exception as e:
                    # Show error dialog
                    error_dialog = QMessageBox()
                    error_dialog.setWindowTitle("Model Import Failed")
                    error_dialog.setText(f"Importing Models Failed: {e}")
                    error_dialog.setIcon(QMessageBox.Critical)
                    error_dialog.exec()
                
                # Save the selected directory to app settings
                self.app_settings.model_path = selected_dir

    def exec(self) -> Tuple[AppSettings, OCRSettings, List[OCRModel]]:
        result = super().exec()

        if result == QDialog.DialogCode.Accepted:
            # update app settings
            self.app_settings.theme = Theme(self.theme_group.checkedId())

            for btn in self.language_buttons:
                if btn.isChecked():
                    self.app_settings.language = Language(
                        self.language_group.id(btn)
                    )
                    break

            # update ocr settings
            for btn in self.line_mode_buttons:
                if btn.isChecked():
                    self.ocr_settings.line_mode = LineMode(
                        self.line_mode_group.id(btn)
                    )
                    break

            for btn in self.encoding_buttons:
                if btn.isChecked():
                    self.ocr_settings.output_encoding = Encoding(
                        self.encodings_group.id(btn)
                    )
                    break

            for btn in self.dewarp_buttons:
                if btn.isChecked():
                    self.ocr_settings.dewarping = self.dewarp_group.id(btn) == 1
                    break

            for btn in self.merge_buttons:
                if btn.isChecked():
                    self.ocr_settings.merge_lines = self.merge_group.id(btn) == 1
                    break

            try:
                self.ocr_settings.k_factor = float(self.k_factor_edit.text())
            except ValueError:
                pass

            try:
                self.ocr_settings.bbox_tolerance = float(
                    self.bbox_tolerance_edit.text()
                )
            except ValueError:
                pass

            return self.app_settings, self.ocr_settings, self.ocr_models

        return self.app_settings, self.ocr_settings, self.ocr_models
