from typing import cast
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QWidget
)

from BDRC.Data import *
from BDRC.MVVM.viewmodel import SettingsViewModel
from BDRC.Widgets.Dialogs import NotificationDialog
from BDRC.Widgets.Dialogs.helpers import build_encodings, build_export_formats, build_file_mode_settings
from BDRC.Widgets.Dialogs.export_dir_or_file_dialog import ExportDirDialog, ExportFileDialog
from BDRC.Exporter import Exporter
from BDRC.Widgets.Utils.HLine import HLine


class ExportDialog(QDialog):
    def __init__(
            self,
            ocr_data: List[OCRData],
            settingsview_model: SettingsViewModel
        ):
        super().__init__()
        self.setObjectName("ExportDialog")

        self.ocr_data = ocr_data

        self._settingsview_model = settingsview_model
        self.export_settings = self._settingsview_model.get_export_settings()

        self.export_settings.output_dir = self.export_settings.output_dir
        self.export_settings.output_file = self.export_settings.output_file

        self.main_label = QLabel("Export OCR Data")
        self.main_label.setObjectName("OptionsLabel")

        self.hline1 = HLine()

        self.file_mode_group, file_mode_buttons = build_file_mode_settings(self.export_settings.file_mode)
        self.exporter_group, self.exporter_buttons = build_export_formats(self.export_settings.format)
        self.encodings_group, self.encoding_buttons = build_encodings(self.export_settings.encoding)

        # build layout
        self.setWindowTitle("BDRC Export")
        self.setMinimumHeight(220)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Export Directory (for File Per Page mode)
        self.export_dir_label = QLabel("Output directory")
        self.export_dir_label.setObjectName("OptionsLabel")

        self.export_dir_layout = QHBoxLayout()
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setObjectName("")
        self.export_dir_edit.setText(self.export_settings.output_dir)

        self.export_dir_select_btn = QPushButton("Select")
        self.export_dir_select_btn.setObjectName("SmallDialogButton")
        self.export_dir_layout.addWidget(self.export_dir_label)
        self.export_dir_layout.addWidget(self.export_dir_edit)
        self.export_dir_layout.addWidget(self.export_dir_select_btn)

        # Export File (for One Big File mode)
        self.export_file_label = QLabel("Output file")
        self.export_file_label.setObjectName("OptionsLabel")

        self.export_file_layout = QHBoxLayout()
        self.export_file_edit = QLineEdit()
        self.export_file_edit.setObjectName("")
        self.export_file_edit.setText(self.export_settings.output_file)

        self.export_file_select_btn = QPushButton("Select")
        self.export_file_select_btn.setObjectName("SmallDialogButton")
        self.export_file_layout.addWidget(self.export_file_label)
        self.export_file_layout.addWidget(self.export_file_edit)
        self.export_file_layout.addWidget(self.export_file_select_btn)

        # File Mode
        file_mode_layout = QHBoxLayout()
        file_mode_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in file_mode_buttons:
            file_mode_layout.addWidget(btn)

        # File Mode Explanations
        def make_explanation(text: str):
            expl = QLabel(text)
            expl.setWordWrap(True)
            expl.setObjectName("OptionsExplanation")
            return expl

        self.file_mode_expl = make_explanation("""
          The <b>file per page</b> mode writes one output file per input page into the selected output directory.
          The output file is named after the input. For example, given two image files <b>one.jpg</b> and <b>two.jpg</b>, the text
          export would write two files <b>one.txt</b> and <b>two.txt</b> into the output directory.

          <p>The <b>one big file</b> mode writes one <i>combined</i> output file containing <i>all</i> input pages.
          For example, given two image files <b>one.jpg</b> and <b>two.jpg</b>, the text export would write one output
          file that has the given file name and contains the OCR text from both images.
        """)

        # Encoding
        encoding_layout = QHBoxLayout()
        encoding_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.encoding_buttons:
            encoding_layout.addWidget(btn)

        # Export Formats
        exporter_layout = QHBoxLayout()
        exporter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.exporter_buttons:
            exporter_layout.addWidget(btn)

        self.hline2 = HLine()

        # Before Page Text
        self.before_page_text_label = QLabel("Before page text")
        self.before_page_text_label.setObjectName("OptionsLabel")
        self.before_page_text_edit = QLineEdit()
        self.before_page_text_edit.setObjectName("")
        self.before_page_text_edit.setText(self.export_settings.before_page_text)

        self.before_page_h_layout = QHBoxLayout()
        self.before_page_h_layout.addWidget(self.before_page_text_label)
        self.before_page_h_layout.addWidget(self.before_page_text_edit)

        # After Page Text
        self.after_page_text_label = QLabel("After page text")
        self.after_page_text_label.setObjectName("OptionsLabel")
        self.after_page_text_edit = QLineEdit()
        self.after_page_text_edit.setObjectName("")
        self.after_page_text_edit.setText(self.export_settings.after_page_text)

        self.after_page_h_layout = QHBoxLayout()
        self.after_page_h_layout.addWidget(self.after_page_text_label)
        self.after_page_h_layout.addWidget(self.after_page_text_edit)

        # Make before and after labels the same width
        beforeafter_label_w = 120
        self.before_page_text_label.setMinimumWidth(beforeafter_label_w)
        self.after_page_text_label.setMinimumWidth(beforeafter_label_w)

        # Before & After Page Text Explanation
        self.before_and_after_page_text_expl = make_explanation("""
            <style>
            ul b {
              font-style: normal;
            }
            </style>

            The <b>before and after page texts</b> will be added before/after every page in the output.<br>
            Several variables are supported:
            <ul>
            <li><b>\\n</b> - Inserts a line break</li>
            <li><b>{image}</b> - Inserts the page image's filename (without file extension, e.g. .jpg)</li>
            </ul>
        """)

        # Export & Cancel buttons
        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Export")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)

        # bind signals
        self.ok_btn.clicked.connect(self.export)
        self.cancel_btn.clicked.connect(self.cancel)
        self.export_dir_select_btn.clicked.connect(self.select_export_dir)
        self.export_file_select_btn.clicked.connect(self.select_export_file)
        self.file_mode_group.buttonClicked.connect(self.show_only_relevant_widgets)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.main_label)
        self.v_layout.addLayout(encoding_layout)
        self.v_layout.addLayout(exporter_layout)

        self.v_layout.addWidget(self.hline1)

        self.v_layout.addLayout(file_mode_layout)
        self.v_layout.addWidget(self.file_mode_expl)

        # Only for 'file per page' mode
        self.v_layout.addLayout(self.export_dir_layout)
        # Only for 'one big file' mode
        self.v_layout.addLayout(self.export_file_layout)

        self.v_layout.addWidget(self.hline2)

        self.v_layout.addLayout(self.before_page_h_layout)
        self.v_layout.addLayout(self.after_page_h_layout)
        self.v_layout.addWidget(self.before_and_after_page_text_expl)

        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        self.widgets_for_file_mode = {
            ExportFileMode.FilePerPage:
                [self.export_dir_label, self.export_dir_edit, self.export_dir_select_btn],
            ExportFileMode.OneCombinedFile:
                [self.export_file_label, self.export_file_edit, self.export_file_select_btn],
        }

        self.show_only_relevant_widgets(None)

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;

            QLabel {
                color: #000000;
            }
            """
        )

    def export(self):
        # Re-populate self.export_settings with settings from widgets
        self.update_export_settings()

        # Validation
        if self.export_settings.file_mode == ExportFileMode.FilePerPage and self.export_settings.output_dir == "":
            NotificationDialog("Invalid settings", "Output directory must be set when 'file per page' is selected").exec()
            return
        if self.export_settings.file_mode == ExportFileMode.OneCombinedFile and self.export_settings.output_file == "":
            NotificationDialog("Invalid settings", "Output file must be set when 'one big file' is selected").exec()
            return

        # Store new export settings to disk
        self.save_export_settings()

        # create exporter based on selection
        exporter = Exporter.create_exporter(self.export_settings)

        # export all data
        print(f"Exporting OCR data for {len(self.ocr_data)} images")
        exporter.export(self.ocr_data)

        self.accept()

        NotificationDialog("Export complete", f"Export of {len(self.ocr_data)} pages successful").exec()

    def cancel(self):
        self.reject()

    def select_export_dir(self):
        _dialog = ExportDirDialog()
        selected_dir = _dialog.exec()

        if selected_dir == 1:
            _selected_dir = _dialog.selectedFiles()[0]
            self.export_dir_edit.setText(_selected_dir)

    def select_export_file(self):
        _dialog = ExportFileDialog()
        selected_file = _dialog.exec()

        if selected_file == 1:
            _selected_file = _dialog.selectedFiles()[0]
            self.export_file_edit.setText(_selected_file)

    def update_export_settings(self):
        self.export_settings = ExportSettings(
            file_mode=ExportFileMode(self.file_mode_group.checkedId()),
            format=ExportFormat(self.exporter_group.checkedId()),
            encoding=Encoding(self.encodings_group.checkedId()),
            output_dir=self.export_dir_edit.text(),
            output_file=self.export_file_edit.text(),
            before_page_text=self.before_page_text_edit.text(),
            after_page_text=self.after_page_text_edit.text()
        )

    def save_export_settings(self):
        self._settingsview_model.save_export_settings(self.export_settings)

    def show_only_relevant_widgets(self, _):
        active_file_mode = ExportFileMode(self.file_mode_group.checkedId())
        for (file_mode, widgets) in self.widgets_for_file_mode.items():
            show = file_mode == active_file_mode
            for widget in widgets:
                cast(QWidget, widget).setVisible(show)
