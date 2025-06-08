from typing import List
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QButtonGroup,
    QProgressBar,
    QComboBox
)

from BDRC.Data import OCRData, OCRModel, OCRSettings, OCRSample, OCResult, Encoding
from BDRC.Inference import OCRPipeline
from BDRC.Runner import OCRBatchRunner
from BDRC.Widgets.Dialogs.helpers import build_encodings, build_binary_selection, build_export_formats

class BatchOCRDialog(QDialog):
    sign_ocr_result = Signal(OCResult)
    last_selected_model_index = 0

    def __init__(
        self,
        data: List[OCRData],
        ocr_pipeline: OCRPipeline,
        ocr_models: List[OCRModel],
        ocr_settings: OCRSettings,
        threadpool: QThreadPool,
        current_model: OCRModel = None,
    ):
        super().__init__()
        self.setObjectName("BatchOCRDialog")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.runner = None
        self.processing = False
        self.data = data
        self.pipeline = ocr_pipeline
        self.ocr_models = ocr_models
        self.ocr_settings = ocr_settings
        self.threadpool = threadpool
        self.current_model = current_model
        
        self.setWindowTitle("Batch Process")
        self.setMinimumWidth(600)
        self.setMaximumWidth(1200)
        self.setFixedHeight(340)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Initialize progress bar in a stopped state
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("DialogProgressBar")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.data))  # Set to number of images
        self.progress_bar.setValue(0)  # Start at 0

        self.start_process_btn = QPushButton("Start")
        self.start_process_btn.setObjectName("SmallDialogButton")
        self.cancel_process_btn = QPushButton("Cancel")
        self.cancel_process_btn.setObjectName("SmallDialogButton")
        self.cancel_process_btn.setEnabled(False)  # Disabled until processing starts

        # settings elements
        # Exports
        # TODO: Not yet used
        # self.exporter_group, self.exporter_buttons = build_exporter_settings()
        self.encodings_group, self.encoding_buttons = build_encodings(
            self.ocr_settings.output_encoding
        )
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(
            self.ocr_settings.dewarping
        )
        self.merge_group, self.merge_buttons = build_binary_selection(
            self.ocr_settings.merge_lines
        )

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
        self.label.setStyleSheet(
            """
            font-weight: bold;
        """
        )

        self.model_selection = QComboBox()
        self.model_selection.setStyleSheet(
            """
                color: #ffffff;
                background: #434343;
                border: 2px solid #ced4da;
                border-radius: 4px;
            """
        )

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            # Temporarily block signals to avoid triggering update during setup
            self.model_selection.blockSignals(True)
            
            # Add models to the dropdown
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)
            
            # Determine which model to select
            selected_index = 0
            
            # If a current model is provided, select it
            if self.current_model is not None:
                for i, model in enumerate(self.ocr_models):
                    if model.config == self.current_model.config:
                        selected_index = i
                        break
            # Otherwise use the remembered index if it's valid
            elif BatchOCRDialog.last_selected_model_index < len(self.ocr_models):
                selected_index = BatchOCRDialog.last_selected_model_index
            
            # Set the selection
            self.model_selection.setCurrentIndex(selected_index)
            
            # Re-enable signals
            self.model_selection.blockSignals(False)

        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        self.ocr_settings_layout = QVBoxLayout()

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

        # merging ocr_lines
        merge_label = QLabel("Merge Lines")
        merge_label.setObjectName("OptionsLabel")
        merge_layout = QHBoxLayout()
        merge_layout.addWidget(merge_label)

        for btn in self.merge_buttons:
            merge_layout.addWidget(btn)

        # other settings
        other_settings_layout = QHBoxLayout()
        k_factor_label = QLabel("K-factor")
        k_factor_label.setObjectName("OptionsLabel")
        self.k_factor_edit = QLineEdit()
        self.k_factor_edit.setText(str(self.ocr_settings.k_factor))
        self.k_factor_edit.editingFinished.connect(self.validate_kfactor_input)

        bbox_tolerance_label = QLabel("Bbox tolerance")
        bbox_tolerance_label.setObjectName("OptionsLabel")
        self.bbox_tolerance_edit = QLineEdit()
        self.bbox_tolerance_edit.setText(str(self.ocr_settings.bbox_tolerance))
        self.bbox_tolerance_edit.editingFinished.connect(
            self.validate_bbox_tolerance_input
        )

        spacer = QLabel()
        spacer.setFixedWidth(60)
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

        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Status")
        self.status_label.setObjectName("OptionsLabel")
        self.status = QLabel("Ready")  # Set initial status
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

    def start_process(self):
        if self.processing:
            return
            
        self.processing = True
        self.status.setText("Processing...")
        self.progress_bar.setValue(0)
        self.start_process_btn.setEnabled(False)
        self.cancel_process_btn.setEnabled(True)
        
        # Create runner
        self.runner = OCRBatchRunner(
            data=self.data,
            ocr_pipeline=self.pipeline,
            mode=self.ocr_settings.line_mode,
            dewarp=self.ocr_settings.dewarping,
            merge_lines=self.ocr_settings.merge_lines,
            k_factor=self.ocr_settings.k_factor,
            bbox_tolerance=self.ocr_settings.bbox_tolerance,
            target_encoding=self.ocr_settings.output_encoding
        )
        
        # Connect signals
        self.runner.signals.sample.connect(self.handle_sample)
        self.runner.signals.error.connect(self.handle_error)
        self.runner.signals.finished.connect(self.handle_finished)
        self.runner.signals.ocr_result.connect(self.handle_ocr_result)
        
        # Start processing
        self.threadpool.start(self.runner)

    def handle_sample(self, sample: OCRSample):
        self.progress_bar.setValue(sample.cnt + 1)  # Add 1 since cnt is 0-based
        self.status.setText(f"Processing {sample.name}")

    def handle_error(self, error_msg: str):
        self.status.setText(f"Error: {error_msg}")
        self.status.setStyleSheet(
            """
                background-color: #A40021;
            """
        )
        self.processing = False
        self.start_process_btn.setEnabled(True)
        self.cancel_process_btn.setEnabled(False)

    def handle_ocr_result(self, result: OCResult):
        # Forward the OCR result to any connected slots
        self.sign_ocr_result.emit(result)

    def handle_finished(self):
        if not self.processing:  # If we were cancelled
            return
            
        self.status.setText("Completed")
        self.status.setStyleSheet(
            """
                background-color: #003d66;
            """
        )
        self.processing = False
        self.start_process_btn.setEnabled(True)
        self.cancel_process_btn.setEnabled(False)
        self.runner = None

    def cancel_process(self):
        if self.runner and self.processing:
            self.runner.kill()
            self.status.setText("Cancelled")
            self.status.setStyleSheet(
                """
                    background-color: #A40021;
                """
            )
            self.processing = False
            self.start_process_btn.setEnabled(True)
            self.cancel_process_btn.setEnabled(False)
            self.runner = None

    def closeEvent(self, event):
        self.cancel_process()
        super().closeEvent(event)

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
        # Update pipeline with selected model
        self.pipeline.update_ocr_model(self.ocr_models[index].config)
        
        # Remember the selection for future dialog instances
        BatchOCRDialog.last_selected_model_index = index
