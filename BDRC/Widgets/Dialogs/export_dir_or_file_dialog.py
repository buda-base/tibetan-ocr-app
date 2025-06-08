from PySide6.QtWidgets import QFileDialog

class ExportDirDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ExportDirDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.Directory)

class ExportFileDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ExportFileDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.AnyFile)
