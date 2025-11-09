import re

from typing import List
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QButtonGroup,
    QCheckBox
)

from BDRC.Data import OCRData, Encoding, OCRLine
from BDRC.Widgets.Dialogs.helpers import build_encodings, build_exporter_settings
from BDRC.Widgets.Dialogs.export_dir_dialog import ExportDirDialog
from BDRC.Exporter import PageXMLExporter, JsonExporter, TextExporter

class ExportDialog(QDialog):
    last_export_dir = None  # Remember last export folder for this session
    SETTINGS_ORG = "BDRC"
    SETTINGS_APP = "TibetanOCRApp"
    SETTINGS_EXPORT_DIR = "export_dialog/last_export_dir"
    SETTINGS_ENCODING = "export_dialog/encoding"
    SETTINGS_SINGLE_FILE = "export_dialog/single_file"
    SETTINGS_INSERT_PAGE_NUM = "export_dialog/insert_page_number"

    def __init__(
            self,
            ocr_data: List[OCRData],
            active_encoding: Encoding,
        ):
        super().__init__()
        self.setObjectName("ExportDialog")
        self.ocr_data = ocr_data
        self.encoding = active_encoding
        import os
        from pathlib import Path
        # Use last export dir if set, otherwise default to user's Downloads folder
        # Load settings
        settings = QSettings(ExportDialog.SETTINGS_ORG, ExportDialog.SETTINGS_APP)
        last_dir = settings.value(ExportDialog.SETTINGS_EXPORT_DIR, None)
        single_file_checked = settings.value(ExportDialog.SETTINGS_SINGLE_FILE, False, type=bool)
        insert_page_number_checked = settings.value(ExportDialog.SETTINGS_INSERT_PAGE_NUM, False, type=bool)

        if last_dir:
            self.output_dir = last_dir
        elif ExportDialog.last_export_dir:
            self.output_dir = ExportDialog.last_export_dir
        else:
            downloads = os.path.join(Path.home(), "Downloads")
            self.output_dir = downloads if os.path.isdir(downloads) else str(Path.home())
        self.main_label = QLabel("Export OCR Data")
        self.main_label.setObjectName("OptionsLabel")
        self.exporter_group, self.exporter_buttons = build_exporter_settings()
        # hide exporter options (only TXT supported for now)
        for btn in self.exporter_buttons:
            btn.hide()
        self.encodings_group, self.encoding_buttons = build_encodings(self.encoding)

        # Restore encoding selection from settings if available
        encoding_id = settings.value(ExportDialog.SETTINGS_ENCODING, None)
        if encoding_id is not None:
            encoding_id = int(encoding_id)
            for btn in self.encoding_buttons:
                if btn.property('encoding_id') == encoding_id:
                    btn.setChecked(True)
                    break
            self.encodings_group.setId(self.encoding_buttons[self.encoding_buttons.index(btn)], encoding_id)
        # Otherwise, keep the default provided

        # Add checkboxes for export options
        self.single_file_checkbox = QCheckBox("Export all as a single file")
        self.single_file_checkbox.setChecked(single_file_checked)
        self.page_number_checkbox = QCheckBox("Insert page number before each page")
        self.page_number_checkbox.setChecked(insert_page_number_checked)
        self.page_number_checkbox.setEnabled(single_file_checked)
        self.single_file_checkbox.stateChanged.connect(self.toggle_page_number_checkbox)

        # build layout
        self.setWindowTitle("BDRC Export")
        self.setMinimumHeight(220)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.export_dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setObjectName("")
        self.dir_edit.setText(self.output_dir)

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
        self.v_layout.addWidget(self.single_file_checkbox)
        self.v_layout.addWidget(self.page_number_checkbox)
        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;

            QLabel {
                color: #ffffff;
            }

            QCheckBox {
                color: #ffffff;
            }
            """
        )

    def export(self):
        # Use directory from UI input (handle manual path entries)
        self.output_dir = self.dir_edit.text().strip()
        # Normalize path separators for the OS
        import os
        self.output_dir = os.path.normpath(self.output_dir)
        if not self.output_dir:
            return

        selected_id = self.exporter_group.checkedId()
        selected_encoding_id = self.encodings_group.checkedId()
        export_single_file = self.single_file_checkbox.isChecked()
        insert_page_numbers = self.page_number_checkbox.isChecked()

        # create exporter based on selection
        if selected_id == 0:
            exporter = TextExporter(self.output_dir)
        elif selected_id == 1:
            exporter = PageXMLExporter(self.output_dir)
        elif selected_id == 2:
            exporter = JsonExporter(self.output_dir)
        else:
            exporter = TextExporter(self.output_dir)

        if export_single_file and isinstance(exporter, TextExporter):
            # Combine all pages into a single file
            all_lines = []
            from pyewts import pyewts
            converter = pyewts() if selected_encoding_id == Encoding.Wylie.value else None
            for idx, data in enumerate(self.ocr_data, 1):
                if data.ocr_lines is not None:
                    lines = data.ocr_lines
                    if selected_encoding_id == Encoding.Wylie.value:
                        lines = [OCRLine(l.guid, converter.toWylie(l.text), Encoding.Wylie) for l in data.ocr_lines]
                    if insert_page_numbers:
                        all_lines.append(OCRLine(None, f"--- Page {idx} ---", None))
                    all_lines.extend(lines)
            # Export to a single file named after the original file, with .txt extension
            import os
            def get_original_basename(name):
                base = os.path.splitext(os.path.basename(name))[0]
                return re.sub(r' - page \d+$', '', base)
            if self.ocr_data and hasattr(self.ocr_data[0], 'image_name'):
                base = get_original_basename(self.ocr_data[0].image_name)
                export_filename = base
            else:
                export_filename = "exported_all"
            exporter.export_text(export_filename, all_lines)
        else:
            # Default: per-page export
            for data in self.ocr_data:
                if isinstance(exporter, TextExporter):
                    if data.ocr_lines is not None:
                        if selected_encoding_id == Encoding.Wylie.value:
                            from pyewts import pyewts
                            converter = pyewts()
                            wylie_lines = []
                            for l in data.ocr_lines:
                                wylie_text = converter.toWylie(l.text)
                                wylie_lines.append(OCRLine(l.guid, wylie_text, Encoding.Wylie))
                            exporter.export_text(data.image_name, wylie_lines)
                        else:
                            exporter.export_text(data.image_name, data.ocr_lines)
                else:
                    exporter.export(data, selected_encoding_id)

        # Save settings before closing
        settings = QSettings(ExportDialog.SETTINGS_ORG, ExportDialog.SETTINGS_APP)
        settings.setValue(ExportDialog.SETTINGS_EXPORT_DIR, self.output_dir)
        settings.setValue(ExportDialog.SETTINGS_ENCODING, self.encodings_group.checkedId())
        settings.setValue(ExportDialog.SETTINGS_SINGLE_FILE, self.single_file_checkbox.isChecked())
        settings.setValue(ExportDialog.SETTINGS_INSERT_PAGE_NUM, self.page_number_checkbox.isChecked())
        self.accept()

    def cancel(self):
        self.reject()

    def select_export_dir(self):
        _dialog = ExportDirDialog()
        selected_dir = _dialog.exec()

        if selected_dir == 1:
            _selected_dir = _dialog.selectedFiles()[0]
            self.output_dir = _selected_dir
            ExportDialog.last_export_dir = _selected_dir  # Remember for next time
            self.dir_edit.setText(self.output_dir)
            # Persist the new directory immediately
            settings = QSettings(ExportDialog.SETTINGS_ORG, ExportDialog.SETTINGS_APP)
            settings.setValue(ExportDialog.SETTINGS_EXPORT_DIR, self.output_dir)

    def toggle_page_number_checkbox(self, state):
        self.page_number_checkbox.setEnabled(bool(state))
        if not state:
            self.page_number_checkbox.setChecked(False)
