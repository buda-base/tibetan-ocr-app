from uuid import UUID
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

from BDRC.Data import AppSettings, OCRSettings, OCRModel, LineMode, Encoding
from BDRC.Utils import import_local_models
from BDRC.Translation import tr
from BDRC.Widgets.Dialogs.helpers import (
    build_line_mode,
    build_encodings,
    build_languages,
    build_binary_selection
)

class SettingsDialog(QDialog):
    SETTINGS_ORG = "BDRC"
    SETTINGS_APP = "TibetanOCRApp"

    SETTINGS_LINE_MODE = "settings_dialog/line_mode"
    SETTINGS_ENCODING = "settings_dialog/encoding"
    SETTINGS_DEWARP = "settings_dialog/dewarp"
    SETTINGS_MERGE = "settings_dialog/merge"
    SETTINGS_K_FACTOR = "settings_dialog/k_factor"
    SETTINGS_BBOX_TOL = "settings_dialog/bbox_tolerance"
    SETTINGS_MODEL_PATH = "settings_dialog/model_path"

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
        self.selected_exporters = []

        # Load settings
        from PySide6.QtCore import QSettings
        settings = QSettings(SettingsDialog.SETTINGS_ORG, SettingsDialog.SETTINGS_APP)
        line_mode_val = settings.value(SettingsDialog.SETTINGS_LINE_MODE, None)
        encoding_val = settings.value(SettingsDialog.SETTINGS_ENCODING, None)
        dewarp_val = settings.value(SettingsDialog.SETTINGS_DEWARP, None)
        merge_val = settings.value(SettingsDialog.SETTINGS_MERGE, None)
        k_factor_val = settings.value(SettingsDialog.SETTINGS_K_FACTOR, None)
        bbox_tol_val = settings.value(SettingsDialog.SETTINGS_BBOX_TOL, None)
        model_path_val = settings.value(SettingsDialog.SETTINGS_MODEL_PATH, None)

        self.import_models_btn = QPushButton("Import Models")
        self.import_models_btn.setObjectName("SmallDialogButton")
        self.import_models_btn.clicked.connect(self.handle_model_import)

        # Restore line mode
        line_mode = self.ocr_settings.line_mode
        if line_mode_val is not None:
            try:
                line_mode = LineMode(int(line_mode_val))
            except Exception:
                pass
        self.line_mode_group, self.line_mode_buttons = build_line_mode(line_mode)

        # Ensure only one line mode button is checked
        for btn in self.line_mode_buttons:
            btn.setChecked(self.line_mode_group.id(btn) == line_mode.value)

        # Restore encoding
        encoding = self.app_settings.encoding
        if encoding_val is not None:
            try:
                encoding = Encoding(int(encoding_val))
            except Exception:
                pass
        self.encodings_group, self.encoding_buttons = build_encodings(encoding)

        # Ensure only one encoding button is checked
        for btn in self.encoding_buttons:
            btn.setChecked(self.encodings_group.id(btn) == encoding.value)

        # Restore dewarping
        dewarp = self.ocr_settings.dewarping
        if dewarp_val is not None:
            dewarp = bool(int(dewarp_val)) if isinstance(dewarp_val, str) else bool(dewarp_val)
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(dewarp)

        # Restore merge lines
        merge = self.ocr_settings.merge_lines
        if merge_val is not None:
            merge = bool(int(merge_val)) if isinstance(merge_val, str) else bool(merge_val)
        self.merge_group, self.merge_buttons = build_binary_selection(merge)

        # Restore k-factor
        if k_factor_val is not None:
            try:
                self.ocr_settings.k_factor = float(k_factor_val)
            except Exception:
                pass
        if hasattr(self, 'k_factor_edit'):
            self.k_factor_edit.setText(str(self.ocr_settings.k_factor))

        # Restore bbox tolerance
        if bbox_tol_val is not None:
            try:
                self.ocr_settings.bbox_tolerance = float(bbox_tol_val)
            except Exception:
                pass
        if hasattr(self, 'bbox_tolerance_edit'):
            self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))

        # Restore model path
        if model_path_val is not None:
            self.app_settings.model_path = model_path_val

        self.setWindowTitle("BDRC Settings")
        self.setMinimumHeight(460)
        self.setMinimumWidth(800)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # define layout
        self.settings_tabs = QTabWidget()


        # OCR Models Tab
        self.ocr_models_tab = QWidget()

        self.data_table = QTableWidget()
        self.data_table.setObjectName("ModelTable")
        self.data_table.setColumnCount(5)
        self.data_tabel_header = [
            "Model",
            "Encoding",
            "Architecture",
            "Version",
            "Model file",
        ]
        self.data_table.setAutoScroll(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)

        self.ocr_label = QLabel("Available OCR Models")
        self.ocr_label.setObjectName("OptionsLabel")

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.ocr_label)
        h_layout.addWidget(self.import_models_btn)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.data_table)
        self.ocr_models_tab.setLayout(v_layout)

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
        encoding_label = QLabel(tr("Encoding"))
        encoding_label.setObjectName("OptionsLabel")
        encoding_layout.addWidget(encoding_label)

        for encoding in self.encoding_buttons:
            encoding_layout.addWidget(encoding)

        # dewarping
        dewarping_layout = QHBoxLayout()
        dewarping_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        dewarping_label = QLabel(tr("Dewarping"))
        dewarping_label.setObjectName("OptionsLabel")
        dewarping_layout.addWidget(dewarping_label)
        for btn in self.dewarp_buttons:
            dewarping_layout.addWidget(btn)

        # merge lines
        merge_layout = QHBoxLayout()
        merge_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        merge_label = QLabel(tr("Merge"))
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

        k_factor_explanation = QLabel(
            "The <b>k Factor</b> controls line extraction intensity. "
            "If OCR results are poor, try increasing or decreasing this value."
        )
        k_factor_explanation.setWordWrap(True)
        k_factor_explanation.setObjectName("OptionsExplanation")

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

        bbox_tolerance_explanation = QLabel(
            "The <b>bbox tolerance</b> defines how much of the initial line detection "
            "(highlighted in orange) is allowed to capture descenders or vowels. "
            "High values may cause issues on layouts with tight spacing."
        )
        bbox_tolerance_explanation.setWordWrap(True)
        bbox_tolerance_explanation.setObjectName("OptionsExplanation")

        line_mode_label.setFixedWidth(160)
        encoding_label.setFixedWidth(160)
        dewarping_label.setFixedWidth(160)
        merge_label.setFixedWidth(160)

        # assemble all layouts
        self.ocr_settings_layout.addLayout(line_mode_layout)
        self.ocr_settings_layout.addLayout(encoding_layout)
        self.ocr_settings_layout.addLayout(dewarping_layout)
        self.ocr_settings_layout.addLayout(merge_layout)
        self.ocr_settings_layout.addLayout(k_factor_layout)
        self.ocr_settings_layout.addWidget(k_factor_explanation)
        self.ocr_settings_layout.addLayout(bbox_tolerance_layout)
        self.ocr_settings_layout.addWidget(bbox_tolerance_explanation)
        self.ocr_settings_tab.setLayout(self.ocr_settings_layout)

        # build entire Layout
        self.settings_tabs.addTab(self.ocr_settings_tab, "OCR Settings")
        self.settings_tabs.addTab(self.ocr_models_tab, "OCR Models")

        self.main_v_layout = QVBoxLayout()
        self.main_v_layout.addWidget(self.settings_tabs)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)
        self.button_h_layout.addWidget(self.reset_btn)
        self.main_v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.main_v_layout)

        # bind signals
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.reset_btn.clicked.connect(self.reset_fields_to_defaults)

        self.update_model_table(self.ocr_models)

    def update_model_table(self, ocr_models: List[OCRModel]):
        self.data_table.clear()
        self.data_table.setRowCount(len(ocr_models))
        self.data_table.setHorizontalHeaderLabels(self.data_tabel_header)

        for i, model in enumerate(ocr_models):
            self.add_ocr_model(i, model)

    def add_ocr_model(self, row_idx: int, ocr_model: OCRModel):
        model_name = QTableWidgetItem(ocr_model.name)
        model_name.setForeground(QColor("#ffffff"))
        self.data_table.setItem(row_idx, 0, model_name)

        model_encoding = QTableWidgetItem(ocr_model.config.encoder.name)
        model_encoding.setForeground(QColor("#ffffff"))
        self.data_table.setItem(row_idx, 1, model_encoding)

        model_arch = QTableWidgetItem(ocr_model.config.architecture.name)
        model_arch.setForeground(QColor("#ffffff"))
        self.data_table.setItem(row_idx, 2, model_arch)

        model_version = QTableWidgetItem(str(ocr_model.config.version))
        model_version.setForeground(QColor("#ffffff"))
        self.data_table.setItem(row_idx, 3, model_version)

        model_file = QTableWidgetItem(ocr_model.path)
        model_file.setForeground(QColor("#ffffff"))
        self.data_table.setItem(row_idx, 4, model_file)

    def validate_bbox_tolerance_input(self):
        try:
            value = float(self.bbox_tolerance_edit.text())
            if not (0.0 <= value <= 10.0):
                self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        except ValueError:
            self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))

    def validate_kfactor_input(self):
        try:
            value = float(self.k_factor_edit.text())
            if not (0.0 <= value <= 10.0):
                self.k_factor_edit.setText(str(self.ocr_settings.k_factor))
        except ValueError:
            self.k_factor_edit.setText(str(self.ocr_settings.k_factor))

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()

    def clear_models(self):
        self.data_table.clear()
        self.data_table.setRowCount(0)

    def reset_fields_to_defaults(self):
        # Set all UI fields and in-memory settings to their default values
        from BDRC.Data import OCRSettings, AppSettings, LineMode, Encoding
        # Set defaults (adjust as appropriate for your app)
        # OCRSettings defaults
        default_line_mode = LineMode.Line
        default_encoding = Encoding.Unicode
        default_dewarping = True
        default_merge_lines = False
        default_k_factor = 2.5
        default_bbox_tolerance = 3.0

        # Update UI radio/button groups
        for btn in self.line_mode_buttons:
            btn.setChecked(self.line_mode_group.id(btn) == default_line_mode.value)
        for btn in self.encoding_buttons:
            btn.setChecked(self.encodings_group.id(btn) == default_encoding.value)
        for btn in self.dewarp_buttons:
            btn.setChecked(self.dewarp_group.id(btn) == int(default_dewarping))
        for btn in self.merge_buttons:
            btn.setChecked(self.merge_group.id(btn) == int(default_merge_lines))

        self.k_factor_edit.setText(str(default_k_factor))
        self.bbox_tolerance_edit.setText(str(default_bbox_tolerance))

        # Update in-memory objects (not persisted until Ok)
        self.ocr_settings.line_mode = default_line_mode
        self.ocr_settings.output_encoding = default_encoding
        self.ocr_settings.dewarping = default_dewarping
        self.ocr_settings.merge_lines = default_merge_lines
        self.ocr_settings.k_factor = default_k_factor
        self.ocr_settings.bbox_tolerance = default_bbox_tolerance
        # Model path is not reset

    def handle_model_import(self):
        # Create a file dialog to select a directory
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_dialog.setWindowTitle(tr("Select Model Directory"))
        
        if file_dialog.exec():
            selected_dir = file_dialog.selectedFiles()[0]
            if os.path.isdir(selected_dir):
                try:
                    # Import models from the selected directory
                    imported_models = import_local_models(selected_dir)
                    
                    # Confirm with the user
                    confirm_dialog = QMessageBox()
                    confirm_dialog.setWindowTitle(tr("Confirm Model Import"))
                    confirm_dialog.setText(tr("Do you want to import the selected models? Existing models will be replaced."))
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
            from PySide6.QtCore import QSettings
            settings = QSettings(SettingsDialog.SETTINGS_ORG, SettingsDialog.SETTINGS_APP)

            # update ocr settings
            for btn in self.line_mode_buttons:
                if btn.isChecked():
                    self.ocr_settings.line_mode = LineMode(
                        self.line_mode_group.id(btn)
                    )
                    settings.setValue(SettingsDialog.SETTINGS_LINE_MODE, self.line_mode_group.id(btn))
                    break

            for btn in self.encoding_buttons:
                if btn.isChecked():
                    self.ocr_settings.output_encoding = Encoding(
                        self.encodings_group.id(btn)
                    )
                    settings.setValue(SettingsDialog.SETTINGS_ENCODING, self.encodings_group.id(btn))
                    break

            for btn in self.dewarp_buttons:
                if btn.isChecked():
                    self.ocr_settings.dewarping = self.dewarp_group.id(btn) == 1
                    settings.setValue(SettingsDialog.SETTINGS_DEWARP, int(self.ocr_settings.dewarping))
                    break

            for btn in self.merge_buttons:
                if btn.isChecked():
                    self.ocr_settings.merge_lines = self.merge_group.id(btn) == 1
                    settings.setValue(SettingsDialog.SETTINGS_MERGE, int(self.ocr_settings.merge_lines))
                    break

            try:
                k_val = float(self.k_factor_edit.text())
                # Clamp k_factor to [0.0, 10.0] range if needed
                if k_val < 0.0:
                    k_val = 0.0
                elif k_val > 10.0:
                    k_val = 10.0
                self.ocr_settings.k_factor = k_val
                self.k_factor_edit.setText(str(k_val))
                settings.setValue(SettingsDialog.SETTINGS_K_FACTOR, str(k_val))
            except ValueError:
                # Do not update k_factor if invalid
                pass

            try:
                bbox_val = float(self.bbox_tolerance_edit.text())
                # Clamp bbox tolerance to [0.0, 10.0] range if needed
                if bbox_val < 0.0:
                    bbox_val = 0.0
                elif bbox_val > 10.0:
                    bbox_val = 10.0
                self.ocr_settings.bbox_tolerance = bbox_val
                self.bbox_tolerance_edit.setText(str(bbox_val))
                settings.setValue(SettingsDialog.SETTINGS_BBOX_TOL, str(bbox_val))
            except ValueError:
                pass

            # Save model path if set
            if hasattr(self.app_settings, 'model_path') and self.app_settings.model_path:
                settings.setValue(SettingsDialog.SETTINGS_MODEL_PATH, self.app_settings.model_path)

            return self.app_settings, self.ocr_settings, self.ocr_models

        return self.app_settings, self.ocr_settings, self.ocr_models
