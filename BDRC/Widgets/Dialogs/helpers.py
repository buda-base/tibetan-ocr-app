from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QRadioButton

from BDRC.Data import LineMode, Language, Encoding
from BDRC.Translation import tr

# Line Models
def build_line_mode(active_mode: LineMode):
    line_mode_group = QButtonGroup()
    line_mode_group.setExclusive(True)
    line_mode_group.setObjectName("OptionsRadio")

    line_btn = QRadioButton(tr("Line"))
    line_btn.setObjectName("OptionsRadio")
    line_btn.setChecked(active_mode == LineMode.Line)
    
    layout_btn = QRadioButton(tr("Layout"))
    layout_btn.setObjectName("OptionsRadio")
    layout_btn.setChecked(active_mode == LineMode.Layout)

    line_mode_group.addButton(line_btn)
    line_mode_group.addButton(layout_btn)
    line_mode_group.setId(line_btn, LineMode.Line.value)
    line_mode_group.setId(layout_btn, LineMode.Layout.value)

    return line_mode_group, [line_btn, layout_btn]

# Languages
def build_languages(active_language: Language):
    language_group = QButtonGroup()
    language_group.setExclusive(True)
    language_group.setObjectName("OptionsRadio")

    tibetan_btn = QRadioButton(tr("Tibetan"))
    tibetan_btn.setObjectName("OptionsRadio")
    tibetan_btn.setChecked(active_language == Language.Tibetan)

    language_group.addButton(tibetan_btn)
    language_group.setId(tibetan_btn, Language.Tibetan.value)

    return language_group, [tibetan_btn]

# Export Formats
def build_exporter_settings():
    exporter_group = QButtonGroup()
    exporter_group.setExclusive(False)
    exporter_group.setObjectName("OptionsRadio")

    txt_btn = QRadioButton(tr("TXT"))
    txt_btn.setObjectName("OptionsRadio")
    txt_btn.setChecked(True)

    exporter_group.addButton(txt_btn)
    exporter_group.setId(txt_btn, 0)

    return exporter_group, [txt_btn]

# Encodings
def build_encodings(active_encoding: Encoding):
    encoding_group = QButtonGroup()
    encoding_group.setExclusive(True)
    encoding_group.setObjectName("OptionsRadio")

    unicode_btn = QRadioButton(tr("Unicode"))
    unicode_btn.setObjectName("OptionsRadio")
    unicode_btn.setChecked(active_encoding == Encoding.Unicode)

    wylie_btn = QRadioButton(tr("Wylie"))
    wylie_btn.setObjectName("OptionsRadio")
    wylie_btn.setChecked(active_encoding == Encoding.Wylie)

    encoding_group.addButton(unicode_btn)
    encoding_group.addButton(wylie_btn)
    encoding_group.setId(unicode_btn, Encoding.Unicode.value)
    encoding_group.setId(wylie_btn, Encoding.Wylie.value)

    return encoding_group, [unicode_btn, wylie_btn]

# Dewarping
def build_binary_selection(current_setting: bool):
    group = QButtonGroup()
    group.setExclusive(True)
    group.setObjectName("OptionsRadio")

    on_btn = QRadioButton(tr("On"))
    on_btn.setObjectName("OptionsRadio")
    on_btn.setChecked(current_setting)

    off_btn = QRadioButton(tr("Off"))
    off_btn.setObjectName("OptionsRadio")
    off_btn.setChecked(not current_setting)

    group.addButton(on_btn)
    group.addButton(off_btn)
    group.setId(on_btn, 1)
    group.setId(off_btn, 0)

    return group, [on_btn, off_btn]
