import os
import cv2
from uuid import UUID
from typing import List, Tuple
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QProgressDialog,
    QPushButton,
    QListWidget,
    QListView,
    QListWidgetItem,
    QWidget,
    QTabWidget,
    QRadioButton,
    QProgressBar,
    QButtonGroup,
    QLineEdit,
    QComboBox
    )

from BDRC.Data import OCRData, OCResult, OCRModel, Theme, AppSettings, OCRSettings, \
    ExportFormat, Language, Encoding, OCRSample
from BDRC.Exporter import PageXMLExporter, JsonExporter, TextExporter
from BDRC.Inference import OCRPipeline
from BDRC.Runner import OCRBatchRunner, OCRunner
from BDRC.Utils import import_local_models
from BDRC.Widgets.Entries import ModelEntryWidget

"""
Boiler plate to construct the Button groups based on the available settings
"""


# Languages
def build_languages(active_language: Language) -> Tuple[QButtonGroup, List[QRadioButton]]:
    buttons = []
    button_group = QButtonGroup()
    button_group.setExclusive(True)

    for lang in Language:
        button = QRadioButton(lang.name)
        button.setObjectName("OptionsRadio")
        buttons.append(button)

        if lang == active_language:
            button.setChecked(True)

        button_group.addButton(button)
        button_group.setId(button, lang.value)

    return button_group, buttons


# Export Formats
def build_exporter_settings() -> Tuple[QButtonGroup, List[QRadioButton]]:
    exporter_buttons = []
    exporters_group = QButtonGroup()
    exporters_group.setExclusive(True)

    for idx, exporter in enumerate(ExportFormat):
        button = QRadioButton(exporter.name)
        button.setObjectName("OptionsRadio")
        exporter_buttons.append(button)

        if idx == 0: # just select the first exporter
            button.setChecked(True)

        exporters_group.addButton(button)
        exporters_group.setId(button, exporter.value)

    return exporters_group, exporter_buttons


# Encodigns
def build_encodings(active_encoding: Encoding) -> Tuple[QButtonGroup, List[QRadioButton]]:
    encoding_buttons = []
    encodings_group = QButtonGroup()
    encodings_group.setExclusive(True)

    for encoding in Encoding:
        button = QRadioButton(encoding.name)
        button.setObjectName("OptionsRadio")

        encoding_buttons.append(button)

        if encoding == active_encoding:
            button.setChecked(True)

        encodings_group.addButton(button)
        encodings_group.setId(button, encoding.value)

    return encodings_group, encoding_buttons


# Dewarping
def build_binary_selection(current_setting: bool) -> Tuple[QButtonGroup, List[QRadioButton]]:
    buttons = []
    button_group = QButtonGroup()
    button_group.setExclusive(True)

    yes_btn = QRadioButton("yes")
    no_btn = QRadioButton("no")
    yes_btn.setObjectName("OptionsRadio")
    no_btn.setObjectName("OptionsRadio")

    if current_setting:
        yes_btn.setChecked(True)
    else:
        no_btn.setChecked(True)

    button_group.addButton(yes_btn)
    button_group.addButton(no_btn)

    button_group.setId(no_btn, 0)
    button_group.setId(yes_btn, 1)

    buttons.append(yes_btn)
    buttons.append(no_btn)

    return button_group, buttons


class ImportImagesDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportImagesDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("Images (*.png *.jpg *.tif *.tiff)")
        self.setViewMode(QFileDialog.ViewMode.List)

class ImportPDFDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportPDFDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilter("PDF file (*.pdf)")
        self.setViewMode(QFileDialog.ViewMode.List)

class ExportDirDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ExportDirDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.Directory)


class ConfirmationDialog(QMessageBox):
    def __init__(self, title: str, message: str, show_cancel: bool = True):
        super().__init__()
        self.setObjectName("ConfirmWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)
        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.cancel_btn = QPushButton("Cancel")

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.ok_btn.setStyleSheet("""
                color: #ffffff;
                font: bold 12px;
                width: 240px;
                height: 32px;
                background-color: #A40021;
                border: 2px solid #A40021;
                border-radius: 4px;

                QPushButton::hover { 
                    color: #ff0000;
                }

            """)

        self.cancel_btn.setStyleSheet("""
                color: #ffffff;
                font: bold 12px; 
                width: 240px;
                height: 32px;
                background-color: #A40021;
                border: 2px solid #A40021;
                border-radius: 4px;

                QPushButton::hover {
                    color: #ff0000;
                }
            """)

        if show_cancel:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)
            self.addButton(self.cancel_btn, QMessageBox.ButtonRole.NoRole)
        else:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)


class NotificationDialog(QMessageBox):
    def __init__(self, title: str, message: str):
        super().__init__()
        self.setObjectName("NotificationWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)

        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setStyleSheet("""
                               color: #ffffff;
                               font: bold 12px;
                               width: 240px;
                               height: 32px;
                               background-color: #A40021;
                               border-radius: 4px;

                               QPushButton::hover { 
                                   color: #ff0000;
                               }
                           """)

        self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)

        self.setStyleSheet("""
                    background-color: #1d1c1c;
                    color: #ffffff;
                    QPushButton {
                        width: 200px;
                        padding: 5px;
                        background-color: #A40021;
                    }
                """)


class ExportDialog(QDialog):
    def __init__(self, ocr_data: List[OCRData], active_exporter: ExportFormat, active_encoding: Encoding):
        super().__init__()
        self.setObjectName("ExportDialog")
        self.ocr_data = ocr_data
        self.exporter = active_exporter
        self.encoding = active_encoding
        self.output_dir = "/"
        self.main_label = QLabel("Export OCR Data")
        self.main_label.setObjectName("OptionsLabel")
        self.exporter_group, self.exporter_buttons = build_exporter_settings(self.exporter)
        self.encodings_group, self.encoding_buttons = build_encodings(self.encoding)

        # build layout
        self.setWindowTitle("BDRC Export")
        self.setMinimumHeight(220)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.export_dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setObjectName("")

        self.dir_select_btn = QPushButton("Select")
        self.dir_select_btn.setObjectName("SmallDialogButton")
        self.export_dir_layout.addWidget(self.dir_edit)
        self.export_dir_layout.addWidget(self.dir_select_btn)

        encoding_layout = QHBoxLayout()
        encoding_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for encoding in self.encoding_buttons:
            encoding_layout.addWidget(encoding)

        export_layout = QHBoxLayout()
        export_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.exporter_buttons:
            export_layout.addWidget(btn)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)

        # bind signals
        self.ok_btn.clicked.connect(self.export)
        self.cancel_btn.clicked.connect(self.cancel)
        self.dir_select_btn.clicked.connect(self.select_export_dir)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.main_label)
        self.v_layout.addLayout(self.export_dir_layout)
        self.v_layout.addLayout(encoding_layout)
        self.v_layout.addLayout(export_layout)
        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;

            QLabel {
                color: #000000;
            }
            """)

    def export(self):
        if os.path.isdir(self.output_dir):
            encoding_id = self.encodings_group.checkedId()
            exporters_id = self.exporter_group.checkedId()
            #converter = pyewts.pyewts()

            _encoding = Encoding(encoding_id)
            _exporter = ExportFormat(exporters_id)

            if _exporter == ExportFormat.XML:
                exporter = PageXMLExporter(self.output_dir)

                for idx, data in enumerate(self.ocr_data):
                    img = cv2.imread(data.image_path)

                    print(f"Exporting OCR text: {len(data.ocr_text)} => {data.ocr_text} for {len(data.lines)} line data")
                    if data.lines is not None and len(data.lines) > 0:

                        exporter.export_lines(
                            image=img,
                            image_name=data.image_name,
                            lines=data.lines,
                            text_lines=data.ocr_text
                        )

            elif _exporter == ExportFormat.JSON:
                exporter = JsonExporter(self.output_dir)

                for idx, data in enumerate(self.ocr_data):
                    img = cv2.imread(data.image_path)

                    if data.lines is not None and len(data.lines) > 0:
                        exporter.export_lines(
                            img,
                            data.image_name,
                            data.lines,
                            data.ocr_text
                        )
            else:
                exporter = TextExporter(self.output_dir)

                for idx, data in enumerate(self.ocr_data):

                    if data.lines is not None and len(data.lines) > 0:
                        exporter.export_text(
                            data.image_name,
                            data.ocr_text
                        )

            self.accept()

        else:
            dialog = NotificationDialog("Invalid Export Directory", "The selected output directory is not valid.")
            dialog.exec()

    def cancel(self):
        self.reject()

    def select_export_dir(self):
        dialog = ExportDirDialog()
        selected_dir = dialog.exec()

        if selected_dir == 1:
            _selected_dir = dialog.selectedFiles()[0]

            if os.path.isdir(_selected_dir):
                self.dir_edit.setText(_selected_dir)
                self.output_dir=_selected_dir
        else:
            note_dialog = NotificationDialog("Invalid Directory", "The selected directory is not valid.")
            note_dialog.exec()


class ModelListWidget(QWidget):
    def __init__(self, guid: UUID, title: str, encoder: str, architecture: str):
        super().__init__()
        self.guid = guid
        self.title = str(title)
        self.encoder = str(encoder)
        self.architecture = str(architecture)

        self.title_label = QLabel(self.title)
        self.encoder_label = QLabel(self.encoder)
        self.architecture_label = QLabel(self.architecture)
        self.download_btn = QPushButton('Download')
        self.delete_btn = QPushButton('Delete')

        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.title_label)
        self.h_layout.addWidget(self.encoder_label)
        self.h_layout.addWidget(self.architecture_label)
        self.h_layout.addWidget(self.download_btn)
        self.h_layout.addWidget(self.delete_btn)
        self.setLayout(self.h_layout)

        self.setStyleSheet("""
            color: #ffffff;
            width: 80%;
        """)



class ModelList(QListWidget):
    sign_on_selected_item = Signal(UUID)

    def __init__(self, parent=None):
        super(ModelList, self).__init__(parent)
        self.parent = parent
        self.setObjectName("ModelList")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)
        self.itemEntered.connect(self.on_item_entered)

        self.setStyleSheet("""
            border 4px solid yellow;
            background-color: #464646;
        
        """)

    def on_item_entered(self, item: QListWidgetItem):
        print(f"Entered Item: {item}")

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(
            item
        )  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, ModelListWidget):
            print(f"Clicked on Model: {_list_item_widget.title}")
            self.sign_on_selected_item.emit(_list_item_widget.guid)


class SettingsDialog(QDialog):
    def __init__(self, app_settings: AppSettings, ocr_settings: OCRSettings, ocr_models: List[OCRModel]):
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

        self.encodings_group, self.encoding_buttons = build_encodings(self.app_settings.encoding)
        self.language_group, self.language_buttons = build_languages(self.app_settings.language)
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(self.ocr_settings.dewarping)
        self.merge_group, self.merge_buttons = build_binary_selection(self.ocr_settings.merge_lines)

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

        self.data_table = QTableWidget()
        self.data_table.setObjectName("ModelTable")
        self.data_table.setColumnCount(5)        
        self.data_tabel_header = ["Model", "Encoding", "Architecture", "Version", "Model file"]
        
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

        # merge lines
        merge_layout = QHBoxLayout()
        merge_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        merge_label = QLabel("Merge")
        merge_label.setObjectName("OptionsLabel")
        merge_layout.addWidget(merge_label)

        for btn in self.merge_buttons:
            merge_layout.addWidget(btn)

        # specific ocr parameters
        spacer = QLabel()
        spacer.setMinimumWidth(200)

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
        k_factor_layout.addWidget(spacer)

        bbox_tolerance_layout = QHBoxLayout()
        bbox_tolerance_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        bbox_tolerance_label = QLabel("Bounding Box Tolerance")
        bbox_tolerance_label.setFixedWidth(160)
        bbox_tolerance_label.setObjectName("OptionsLabel")
        self.bbox_tolerance_edit = QLineEdit()
        self.bbox_tolerance_edit.setObjectName("DialogLineEdit")
        self.bbox_tolerance_edit.setFixedWidth(60)
        self.bbox_tolerance_edit.editingFinished.connect(self.validate_bbox_tolerance_input)
        self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        bbox_tolerance_layout.addWidget(bbox_tolerance_label)
        bbox_tolerance_layout.addWidget(self.bbox_tolerance_edit)
        bbox_tolerance_layout.addWidget(spacer)

        encoding_label.setFixedWidth(160)
        dewarping_label.setFixedWidth(160)
        merge_label.setFixedWidth(160)

        self.ocr_settings_layout.addLayout(encoding_layout)
        self.ocr_settings_layout.addLayout(dewarping_layout)
        self.ocr_settings_layout.addLayout(merge_layout)
        self.ocr_settings_layout.addLayout(k_factor_layout)
        self.ocr_settings_layout.addLayout(bbox_tolerance_layout)
        self.ocr_settings_tab.setLayout(self.ocr_settings_layout)

        # build entire Layout
        #self.settings_tabs.addTab(self.general_settings_tab, "General")
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

        self.setStyleSheet("""
            QPushButton {
                color: #A40021;
                background-color: #fce08d;
                border-radius: 4px;
                height: 18;
            }

            QPushButton::hover {
                color: #ffad00;
            }
        """)

        self.update_model_table(self.ocr_models)

    def update_model_table(self, ocr_models: List[OCRModel]):
        self.data_table.setRowCount(len(ocr_models))
        self.data_table.clear()
        self.data_table.setHorizontalHeaderLabels(self.data_tabel_header)

        for idx, model in enumerate(ocr_models):
            self.add_ocr_model(idx, model)

    def add_ocr_model(self, row_idx: int, ocr_model: OCRModel):

        encoder_widget = QTableWidgetItem(ocr_model.config.encoder.name)
        encoder_widget.setBackground(QColor("#172832"))

        self.data_table.setItem(row_idx, 0, QTableWidgetItem(str(ocr_model.name)))
        self.data_table.setItem(row_idx, 1, encoder_widget)
        self.data_table.setItem(row_idx, 2, QTableWidgetItem(ocr_model.config.architecture.name))
        self.data_table.setItem(row_idx, 3, QTableWidgetItem(str(ocr_model.config.version)))
        self.data_table.setItem(row_idx, 4, QTableWidgetItem(str(ocr_model.path)))
        self.update()

    def validate_bbox_tolerance_input(self):
        try:
            float(self.bbox_tolerance_edit.text())
            self.ocr_settings.bbox_tolerance = float(self.bbox_tolerance_edit.text())

        except ValueError as e:
            print(f"Invalid float value: {e}")
            self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))

    def validate_kfactor_input(self):
        try:
            float(self.k_factor_edit.text())
            self.ocr_settings.k_factor = float(self.k_factor_edit.text())
        except ValueError as e:
            print(f"Invalid float value: {e}")
            self.k_factor_edit.setText(str(self.ocr_settings.k_factor))

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()
        
    def clear_models(self):
       print(f"SettingsDialog -> ClearModels()")

    def handle_model_import(self):
        _dialog = ExportDirDialog()
        selected_dir = _dialog.exec()

        if selected_dir == 1:
            _selected_dir = _dialog.selectedFiles()[0]

            if os.path.isdir(_selected_dir):
                try:
                    imported_models = import_local_models(_selected_dir)

                    confirm_dialog = ConfirmationDialog(
                        title="Confirm Model Import",
                        message="Do you want to import the selected models? Existing models will be replaced."
                    )
                    confirm_dialog.exec()
                    result = confirm_dialog.result()

                    if result == 2:
                        self.ocr_models = imported_models
                        self.update_model_table(self.ocr_models)

                except BaseException as e:
                    error_dialog = NotificationDialog("Model import failed", f"Importing Models Failed: {e}")
                    error_dialog.exec()

            self.app_settings.model_path = _selected_dir

    def exec(self):
        super().exec()

        # fetch settings
        theme_id = self.theme_group.checkedId()
        self.app_settings.theme = Theme(theme_id)

        language_id = self.language_group.checkedId()
        self.app_settings.language = Language(language_id)

        encoding_id = self.encodings_group.checkedId()
        self.app_settings.encoding = Encoding(encoding_id)

        dewarp_id = self.dewarp_group.checkedId()
        do_dewarp = bool(dewarp_id)

        merge_id = self.merge_group.checkedId()
        do_merge = bool(merge_id)

        if self.k_factor_edit.text() != "":
            self.ocr_settings.k_factor = float(self.k_factor_edit.text())

        if self.bbox_tolerance_edit.text() != "":
            self.ocr_settings.bbox_tolerance = float(self.bbox_tolerance_edit.text())

        self.ocr_settings.dewarping = do_dewarp
        self.ocr_settings.merge_lines = do_merge

        return self.app_settings, self.ocr_settings, self.ocr_models


class BatchOCRDialog(QDialog):
    sign_ocr_result = Signal(OCResult)

    def __init__(self,
                 data: List[OCRData],
                 ocr_pipeline: OCRPipeline,
                 ocr_models: List[OCRModel],
                 ocr_settings: OCRSettings,
                 threadpool: QThreadPool
                 ):
        super().__init__()
        self.setObjectName("BatchOCRDialog")
        self.data = data
        self.pipeline = ocr_pipeline
        self.ocr_models = ocr_models
        self.ocr_settings = ocr_settings
        self.threadpool = threadpool
        self.runner = None
        self.output_dir = ""
        self.setWindowTitle("Batch Process")

        self.setMinimumWidth(600)
        self.setMaximumWidth(1200)
        self.setFixedHeight(340)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("DialogProgressBar")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.data)-1)

        self.start_process_btn = QPushButton("Start")
        self.start_process_btn.setObjectName("SmallDialogButton")
        self.cancel_process_btn = QPushButton("Cancel")
        self.cancel_process_btn.setObjectName("SmallDialogButton")

        # settings elements
        # Exports
        self.exporter_group, self.exporter_buttons = build_exporter_settings()
        self.encodings_group, self.encoding_buttons = build_encodings(self.ocr_settings.output_encoding)
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(self.ocr_settings.dewarping)
        self.merge_group, self.merge_buttons = build_binary_selection(self.ocr_settings.merge_lines)

        # build layout
        self.progress_layout = QHBoxLayout()
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.start_process_btn)
        self.progress_layout.addWidget(self.cancel_process_btn)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)

        self.v_layout = QVBoxLayout()
        self.label = QLabel("Batch Processing")
        self.label.setObjectName("OptionsLabel")
        self.label.setStyleSheet("""
            font-weight: bold;
        """)

        self.export_dir_layout = QHBoxLayout()
        self.dir_select_btn = QPushButton("select")
        self.dir_select_btn.setObjectName("SmallDialogButton")

       
        self.model_selection = QComboBox()
        self.model_selection.setStyleSheet("""
                color: #ffffff;
                background: #434343;
                border: 2px solid #ced4da;
                border-radius: 4px;
            """)

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)

        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        self.ocr_settings_layout = QVBoxLayout()

        spacer = QLabel()
        spacer.setFixedWidth(60)
        # encoding
        encoding_label = QLabel("Encoding")
        encoding_label.setObjectName("OptionsLabel")
        encoding_layout = QHBoxLayout()
        encoding_layout.addWidget(encoding_label)

        for btn in self.encoding_buttons:
            encoding_layout.addWidget(btn)
        

        # dewarping
        dewarping_label = QLabel("Dewarping")
        dewarping_label.setObjectName("OptionsLabel")
        dewarping_layout = QHBoxLayout()
        dewarping_layout.addWidget(dewarping_label)

        for btn in self.dewarp_buttons:
            dewarping_layout.addWidget(btn)
 
        # merging lines
        merge_label = QLabel("Merge Lines")
        merge_label.setObjectName("OptionsLabel")
        merge_layout = QHBoxLayout()
        merge_layout.addWidget(merge_label)
        
        for btn in self.merge_buttons:
            merge_layout.addWidget(btn)

        # merging lines
        export_label = QLabel("Export format")
        export_layout = QHBoxLayout()
        export_layout.addWidget(export_label)
        for btn in self.exporter_buttons:
            export_layout.addWidget(btn)

        other_settings_layout = QHBoxLayout()
        other_label = QLabel("Other Settings")
        other_label.setObjectName("OptionsLabel")
        k_factor_label = QLabel("K-factor")
        k_factor_label.setObjectName("OptionsLabel")
        self.k_factor_edit = QLineEdit()
        self.k_factor_edit.setText(str(self.ocr_settings.k_factor))
        self.k_factor_edit.editingFinished.connect(self.validate_kfactor_input)

        bbox_tolerance_label = QLabel("Bbox tolerance")
        bbox_tolerance_label.setObjectName("OptionsLabel")
        self.bbox_tolerance_edit = QLineEdit()
        self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        self.bbox_tolerance_edit.editingFinished.connect(self.validate_bbox_tolerance_input)

        other_settings_layout.addWidget(spacer)
        other_settings_layout.addWidget(k_factor_label)
        other_settings_layout.addWidget(self.k_factor_edit)
        other_settings_layout.addWidget(spacer)
        other_settings_layout.addWidget(bbox_tolerance_label)
        other_settings_layout.addWidget(self.bbox_tolerance_edit)

        # assemble layout
        self.ocr_settings_layout.addLayout(encoding_layout)
        self.ocr_settings_layout.addLayout(dewarping_layout)
        self.ocr_settings_layout.addLayout(merge_layout)
        self.ocr_settings_layout.addLayout(export_layout)

        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Status")
        self.status_label.setObjectName("OptionsLabel")
        self.status = QLabel("")
        self.status.setObjectName("OptionsLabel")
        self.status.setMinimumWidth(180)
        self.status.setFixedHeight(32)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status)

        self.v_layout.addWidget(self.label)
        self.v_layout.addWidget(self.model_selection)
        self.v_layout.addLayout(self.ocr_settings_layout)
        self.v_layout.addLayout(self.progress_layout)
        self.v_layout.addLayout(self.status_layout)
        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        # bind signals
        self.start_process_btn.clicked.connect(self.start_process)
        self.cancel_process_btn.clicked.connect(self.cancel_process)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;
            
            QLineEdit {
                color: #ffffff;
                background-color: #474747;
                border: 2px solid #343942;
                border-radius: 8px;
                padding: 6px;
                text-align: left;
            }
        
        """)

    def validate_bbox_tolerance_input(self):
        try:
            float(self.bbox_tolerance_edit.text())
            self.ocr_settings.bbox_tolerance = float(self.bbox_tolerance_edit.text())

        except ValueError as e:
            print(f"Invalid float value: {e}")
            self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))

    def validate_kfactor_input(self):
        try:
            float(self.k_factor_edit.text())
            self.ocr_settings.k_factor = float(self.k_factor_edit.text())
        except ValueError as e:
            print(f"Invalid float value: {e}")
            self.k_factor_edit.setText(str(self.ocr_settings.k_factor))

    def on_select_ocr_model(self, index: int):
        self.pipeline.update_ocr_model(self.ocr_models[index].config)

    def start_process(self):
        encoding_id = self.encodings_group.checkedId()
        dewarp_id = self.dewarp_group.checkedId()
        merge_id = self.dewarp_group.checkedId()

        encoding = Encoding(encoding_id)
        do_dewarp = bool(dewarp_id)
        do_merge = bool(merge_id)

        k_factor = self.k_factor_edit.text()
        bbox_tolerance = self.bbox_tolerance_edit.text()

        self.runner = OCRBatchRunner(
            self.data,
            self.pipeline,
            dewarp=do_dewarp,
            merge_lines=do_merge,
            k_factor=float(k_factor),
            bbox_tolerance=float(bbox_tolerance)
        )

        self.runner.signals.sample.connect(self.handle_update_progress)
        self.runner.signals.finished.connect(self.finish)
        self.threadpool.start(self.runner)
        self.status.setText("Running")
        self.status.setStyleSheet("""
            background-color: #ff9100;
        """)

    def handle_update_progress(self, sample: OCRSample):
        self.progress_bar.setValue(sample.cnt)
        """ 
        file_name = self.data[sample.cnt].image_name
        out_file = os.path.join(self.output_dir, f"{file_name}.txt")

        with open(out_file, "w", encoding="utf-8") as f:
            for line in sample.result.text:
                f.write(f"{line}\n")
        """
        self.sign_ocr_result.emit(sample.result)

    def finish(self):
        print(f"Thread Completed")
        self.runner = None
        self.status.setText("Finished")
        self.status.setStyleSheet("""
                    background-color: #63ff00;
                """)

    def cancel_process(self):
        if self.runner is not None:
            self.runner.stop = True
            self.status.setText("Canceled")
            self.status.setStyleSheet("""
                                background-color: #e80000;
                            """)


class ImportFilesProgress(QProgressDialog):
    def __init__(self, title: str, max_length: int):
        super(ImportFilesProgress, self).__init__()
        self.setWindowTitle(title)
        self.setFixedWidth(420)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setContentsMargins(32, 32, 32, 32)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.setFixedHeight(32)
        self.setCancelButton(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(max_length)
        self.progress_bar.setObjectName("DialogProgressBar")
        self.progress_bar.setStyleSheet("""
                    QProgressBar {
                        background-color: #474747;
                        color: #A40021;
                        border: 2px solid #fce08e;
                        border-radius: 8px;
                        padding: 4px 4px 4px 4px;
                    }

                    QProgressBar::chunk {
                        background-color: #A40021;
                        width: 20px;
                    }
                """)

        self.setBar(self.progress_bar)


        self.cancel_btn.setStyleSheet("""
            QPushButton {
                color: #ffffff;
                background-color: #A40021;
                border-radius: 4px;
                height: 24;
                margin-right: 24px;
            }
        
        """)

        self.setStyleSheet("""
            color: #ffffff;
            background-color: #1d1c1c;
            QPushButton {
                color: #ffffff;
                background-color: #A40021;
                border-radius: 4px;
                height: 24;
            }
            
            QProgressBar {
                background-color: #24272c;
                border-radius: 5px;
                border-width: 2px;
            }

            QProgressBar::chunk
            {
                background-color: #003d66;
                border-radius: 5px;
                margin: 3px 3px 3px 3px;
            }
            
            """
        )



class OCRDialog(QProgressDialog):
    sign_ocr_result = Signal(OCResult)

    def __init__(self, pipeline: OCRPipeline, settings: OCRSettings, data: OCRData, pool: QThreadPool):
        super(OCRDialog, self).__init__()
        self.setObjectName("OCRDialog")
        self.setMinimumWidth(500)
        self.setWindowTitle("OCR Progress")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimum(0)
        self.setMaximum(0)
        self.pipeline = pipeline
        self.settings = settings
        self.data = data
        self.pool = pool
        self.result = None

        # build layout
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""

                QPushButton {
                    margin-top: 15px;
                    background-color: #ff0000;
                }

                QPushButton::hover {
                    color: #ffad00;
                }

            """)

        self.setCancelButton(self.cancel_btn)
        self.setStyleSheet("""

            background-color: #08081f;

            QProgressBar {
                background-color: #24272c;
                border-radius: 5px;
                border-width: 2px;
            }

            QProgressBar::chunk
            {
                background-color: #003d66;
                border-radius: 5px;
                margin: 3px 3px 3px 3px;
            }
        """)

        self.show()

    def exec(self):
        runner = OCRunner(self.data, self.pipeline, self.settings)
        runner.signals.error.connect(self.handle_error)
        runner.signals.ocr_result.connect(self.handle_ocr_result)
        runner.signals.finished.connect(self.thread_complete)
        self.pool.start(runner)

    def handle_error(self, error: str):
        print(f"Encountered Error: {error}")

    def handle_ocr_result(self, result: OCResult):
        #print(f"Handling ocr result: {result}")
        self.sign_ocr_result.emit(result)

    def thread_complete(self):
        print(f"Thread Complete")
        #self.close()



class TextInputDialog(QDialog):
    def __init__(self, title: str, edit_text: str, parent: None):
        super(TextInputDialog, self).__init__()
        self.parent = parent
        self.title = title
        self.edit_text = edit_text
        self.new_text = ""
        self.setFixedWidth(480)
        self.setWindowTitle(title)
        self.spacer = QLabel()
        self.spacer.setFixedHeight(36)
        self.line_edit = QLineEdit(self)
        self.line_edit.setText(self.edit_text)

        self.line_edit.editingFinished.connect(self.update_text)
        self.line_edit.setStyleSheet("""
            color: #ffffff;
            background-color: #3e5272;
            border: 2px solid #3e5272;
            border-radius: 8px;
            padding: 4px;

        """)

        self.accept_btn = QPushButton("Accept")
        self.reject_btn = QPushButton("Reject")
        self.accept_btn.setObjectName("SmallDialogButton")
        self.reject_btn.setObjectName("SmallDialogButton")

        self.accept_btn.clicked.connect(self.accept_change)
        self.reject_btn.clicked.connect(self.reject_change)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.accept_btn)
        self.h_layout.addWidget(self.reject_btn)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.line_edit)
        self.v_layout.addWidget(self.spacer)
        self.v_layout.addLayout(self.h_layout)

        self.setLayout(self.v_layout)
        self.setStyleSheet("""
            background-color: #08091e;
        """)

    def update_text(self):
        self.new_text = self.line_edit.text()

    def accept_change(self):
        self.accept()

    def reject_change(self):
        self.reject()