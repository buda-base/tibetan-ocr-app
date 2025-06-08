from PySide6.QtWidgets import QButtonGroup, QRadioButton

from BDRC.Data import *

# Line Modes
def build_line_mode(active_mode: LineMode):
    line_mode_group = QButtonGroup()
    line_mode_group.setObjectName("OptionsRadio")

    buttons = []

    for line_mode in LineMode:
        btn = QRadioButton(line_mode.name)
        btn.setObjectName("OptionsRadio")
        btn.setChecked(active_mode == line_mode)

        line_mode_group.addButton(btn)
        line_mode_group.setId(btn, line_mode.value)

        buttons.append(btn)

    return line_mode_group, buttons

# Languages
def build_languages(active_language: Language):
    language_group = QButtonGroup()
    language_group.setExclusive(True)
    language_group.setObjectName("OptionsRadio")

    tibetan_btn = QRadioButton("Tibetan")
    tibetan_btn.setObjectName("OptionsRadio")
    tibetan_btn.setChecked(active_language == Language.Tibetan)

    language_group.addButton(tibetan_btn)
    language_group.setId(tibetan_btn, Language.Tibetan.value)

    return language_group, [tibetan_btn]

# Export Formats
def build_export_formats(active_export_format: ExportFormat):
    exporter_group = QButtonGroup()
    exporter_group.setObjectName("OptionsRadio")

    buttons = []
    for format in ExportFormat:
        btn = QRadioButton(format.name)
        btn.setObjectName("OptionsRadio")
        btn.setChecked(format == active_export_format)

        exporter_group.addButton(btn)
        exporter_group.setId(btn, format.value)

        buttons.append(btn)

    return exporter_group, buttons

# File Modes
def build_file_mode_settings(active_file_mode: ExportFileMode):
    file_mode_group = QButtonGroup()
    file_mode_group.setObjectName("OptionsRadio")

    def make_button(file_mode: ExportFileMode):
        btn = QRadioButton(file_mode.label)
        btn.setObjectName("OptionsRadio")
        btn.setChecked(file_mode == active_file_mode)

        file_mode_group.addButton(btn)
        file_mode_group.setId(btn, file_mode.value)

        return btn

    buttons = (make_button(ExportFileMode.FilePerPage), make_button(ExportFileMode.OneBigFile))

    return file_mode_group, buttons

# Encodings
def build_encodings(active_encoding: Encoding):
    encoding_group = QButtonGroup()
    encoding_group.setExclusive(True)
    encoding_group.setObjectName("OptionsRadio")

    buttons = []

    for encoding in Encoding:
        btn = QRadioButton(encoding.name)
        btn.setObjectName("OptionsRadio")
        btn.setChecked(encoding == active_encoding)

        encoding_group.addButton(btn)
        encoding_group.setId(btn, encoding.value)

        buttons.append(btn)

    return encoding_group, buttons

# Dewarping
def build_binary_selection(current_setting: bool):
    group = QButtonGroup()
    group.setObjectName("OptionsRadio")

    on_btn = QRadioButton("On")
    on_btn.setObjectName("OptionsRadio")
    on_btn.setChecked(current_setting)

    off_btn = QRadioButton("Off")
    off_btn.setObjectName("OptionsRadio")
    off_btn.setChecked(not current_setting)

    group.addButton(on_btn)
    group.addButton(off_btn)
    group.setId(on_btn, 1)
    group.setId(off_btn, 0)

    return group, [on_btn, off_btn]
