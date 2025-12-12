from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import QProgressDialog, QPushButton

from BDRC.data import OCRData, OCResult, OCRSettings
from BDRC.inference import OCRPipeline
from BDRC.runner import OCRunner


class OCRDialog(QProgressDialog):
    sign_ocr_result = Signal(OCResult)

    def __init__(
        self,
        pipeline: OCRPipeline,
        settings: OCRSettings,
        data: OCRData,
        pool: QThreadPool,
    ):
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
        self.start_btn.setObjectName("DialogButton")

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("DialogButton")

        self.setCancelButton(self.cancel_btn)
        self.setStyleSheet(
            """
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
        """
        )

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
        # print(f"Handling ocr result: {result}")
        self.sign_ocr_result.emit(result)

    def thread_complete(self):
        print(f"Thread Complete")
