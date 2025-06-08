import cv2
import sys
import re
import platform
import os
from uuid import UUID
from typing import Dict, List
from PySide6.QtCore import Signal, Qt, QThreadPool
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QMessageBox, QFileDialog
from pdf2image import convert_from_path, pdfinfo_from_path
import Config
from BDRC.Styles import DARK
from BDRC.Inference import OCRPipeline
from BDRC.Data import OpStatus, Platform, OCRData, OCRModel, OCResult
from BDRC.Utils import build_ocr_data, get_filename, create_dir
from BDRC.Widgets.Dialogs import NotificationDialog, ImportFilesProgress, PDFImportDialog, TextInputDialog, ExportDialog, SettingsDialog, BatchOCRDialog
from BDRC.utils.pdf_extract import extract_images_from_pdf
from BDRC.Widgets.Layout import HeaderTools, ImageGallery, Canvas, TextView
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
import logging


class MainView(QWidget):
    s_handle_import = Signal()
    s_handle_pdf_import = Signal()
    s_handle_page_select = Signal(int)
    s_on_file_save = Signal()
    s_run_ocr = Signal(UUID)
    s_run_batch_ocr = Signal()
    s_handle_settings = Signal()

    def __init__(self, data_view: DataViewModel, settings_view: SettingsViewModel, platform: Platform):
        super().__init__()
        self.setObjectName("MainView")
        self.setContentsMargins(0, 0, 0, 0)
        self._data_view = data_view
        self._settings_view = settings_view
        self.platform = platform
        self.resource_dir = self._settings_view.get_execution_dir()
        self.default_font = self._settings_view.get_default_font_path()

        self.header_tools = HeaderTools(self._data_view, self._settings_view)
        self.canvas = Canvas(self.resource_dir )
        self.text_view = TextView(platform=self.platform, dataview=self._data_view, execution_dir=self.resource_dir, font_path=self.default_font)
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setHandleWidth(10)
        self.v_splitter.addWidget(self.canvas)
        self.v_splitter.addWidget(self.text_view)

        # build layout
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.header_tools)
        self.v_layout.addWidget(self.v_splitter)
        self.setLayout(self.v_layout)

        # connect to view model signals
        self._data_view.s_data_selected.connect(self.set_data)
        self._data_view.s_page_data_update.connect(self.set_data)
        self._data_view.s_data_changed.connect(self.update_data)
        self._data_view.s_data_cleared.connect(self.clear_data)


        # connect to tool signals
        self.header_tools.toolbox.s_new.connect(self.handle_new)
        self.header_tools.toolbox.s_import_files.connect(self.handle_import)
        self.header_tools.toolbox.s_import_pdf.connect(self.handle_pdf_import)
        self.header_tools.toolbox.s_save.connect(self.handle_file_save)
        self.header_tools.toolbox.s_on_select_model.connect(self.handle_model_selection)
        self.header_tools.toolbox.s_run.connect(self.handle_run)
        self.header_tools.toolbox.s_run_all.connect(self.handle_batch_run)
        self.header_tools.toolbox.s_settings.connect(self.handle_settings)
        self.header_tools.page_switcher.s_on_page_changed.connect(self.handle_update_page)

        self.current_guid = None

    def handle_new(self):
        self._data_view.clear_data()

    def update_data(self, data: List[OCRData]):
        self.header_tools.update_page_count(len(data))

    def handle_import(self):
        self.s_handle_import.emit()

    def handle_pdf_import(self):
        self.s_handle_pdf_import.emit()

    def handle_file_save(self):
        self.s_on_file_save.emit()

    def handle_model_selection(self, ocr_model: OCRModel):
        self._settings_view.select_ocr_model(ocr_model)

    def handle_run(self):
        if self.current_guid is not None:
            self.s_run_ocr.emit(self.current_guid)
        else:
            dialog = NotificationDialog("No data", "Project contains no data.")
            dialog.exec()

    def handle_batch_run(self):
        self.s_run_batch_ocr.emit()

    def handle_settings(self):
        self.s_handle_settings.emit()

    def handle_update_page(self, index: int):
        self._data_view.select_data_by_index(index)

    def set_data(self, data: OCRData):
        self.canvas.set_preview(data)
        self.current_guid = data.guid

    def clear_data(self):
        self.canvas.clear()


class AppView(QWidget):
    def __init__(self,
                 dataview_model: DataViewModel,
                 settingsview_model: SettingsViewModel,
                 platform: Platform):
        super().__init__()

        appVersion = '.'.join(str(i) for i in Config.APP_VERSION)

        self.setObjectName("MainWindow")
        self.setWindowTitle(f"BDRC OCR [BETA] {appVersion}")
        self.setContentsMargins(0, 0, 0, 0)
        self.platform = platform
        self.threadpool = QThreadPool()
        self._dataview_model = dataview_model
        self._settingsview_model = settingsview_model

        self.tmp_dir = self._settingsview_model.get_tmp_dir()
        self.resource_dir = self._settingsview_model.get_execution_dir()

        self.image_gallery = ImageGallery(self._dataview_model, self.threadpool, self.resource_dir)
        self.main_container = MainView(self._dataview_model, self._settingsview_model, self.platform)
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_splitter.setHandleWidth(10)
        self.h_splitter.setContentsMargins(0, 0, 0, 0)
        self.h_splitter.addWidget(self.image_gallery)
        self.h_splitter.addWidget(self.main_container)

        # build layout
        self.main_layout = QHBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.h_splitter)
        self.setLayout(self.main_layout)

        # connect view model signals
        self._settingsview_model.s_ocr_model_changed.connect(self.update_ocr_model)

        # connect layout signals
        self.main_container.s_handle_import.connect(self.handle_file_import)
        self.main_container.s_handle_pdf_import.connect(self.handle_pdf_import)
        self.main_container.s_handle_page_select.connect(self.select_page)
        self.main_container.s_on_file_save.connect(self.save)
        self.main_container.s_run_ocr.connect(self.run_ocr)
        self.main_container.s_run_batch_ocr.connect(self.run_batch_ocr)
        self.main_container.s_handle_settings.connect(self.handle_settings)

        # ocr inference sessions
        line_config = self._settingsview_model.get_line_model()
        ocr_model = self._settingsview_model.get_current_ocr_model()

        if ocr_model is not None:
            self.ocr_pipeline = OCRPipeline(
                self.platform,
                ocr_model.config,
                line_config)
        else:
            self.ocr_pipeline = None

        # Memoized poppler path
        self._poppler_path = None

        self.show()

    def handle_file_import(self):
        import uuid
        
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;PDF Files (*.pdf);;All Files (*)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            
            if not files:
                return
                
            # Create a temporary directory for imported files
            import_dir = os.path.join(os.path.expanduser("~"), ".bdrc_ocr", "imports")
            create_dir(import_dir)
            
            try:
                results = {}
                
                for file_path in files:
                    file_extension = os.path.splitext(file_path)[1].lower()
                    
                    if file_extension == '.pdf':
                        # Show PDF import options dialog
                        pdf_dialog = PDFImportDialog(self)
                        if pdf_dialog.exec():
                            import_method = pdf_dialog.get_selected_method()
                            
                            # Create a unique directory for this PDF
                            pdf_dir = os.path.join(import_dir, str(uuid.uuid4()))
                            create_dir(pdf_dir)
                            
                            if import_method == PDFImportDialog.IMPORT_EMBEDDED_IMAGES:
                                # Extract embedded images using PyPDF2
                                self.handle_pdf_extract(file_path, pdf_dir, results)
                            else:
                                # Convert pages to images using pdf2image
                                self.convert_pdf_to_images(file_path, pdf_dir, results)
                    else:
                        # Handle regular image files
                        file_id = uuid.uuid4()
                        file_name = get_filename(file_path)
                        
                        data = build_ocr_data(file_id, file_path)
                        results[file_id] = data
                
                if results:
                    self.import_files(results)
                    
            except Exception as e:
                error_dialog = NotificationDialog("Error", f"An error occurred while importing files: {e}")
                error_dialog.exec_()
                
    def handle_pdf_import(self):
        """Handle importing PDF files."""
        import uuid
        
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("PDF Files (*.pdf)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            
            if not files:
                return
                
            # Create a temporary directory for imported files
            import_dir = os.path.join(os.path.expanduser("~"), ".bdrc_ocr", "imports")
            create_dir(import_dir)
            
            try:
                results = {}
                
                for file_path in files:
                    # Show PDF import options dialog
                    pdf_dialog = PDFImportDialog(self)
                    if pdf_dialog.exec():
                        import_method = pdf_dialog.get_selected_method()
                        
                        # Create a unique directory for this PDF
                        pdf_dir = os.path.join(import_dir, str(uuid.uuid4()))
                        create_dir(pdf_dir)
                        
                        if import_method == PDFImportDialog.IMPORT_EMBEDDED_IMAGES:
                            # Extract embedded images using PyPDF2
                            self.handle_pdf_extract(file_path, pdf_dir, results)
                        else:
                            # Convert pages to images using pdf2image
                            self.convert_pdf_to_images(file_path, pdf_dir, results)
                
                if results:
                    self.import_files(results)
                    
            except Exception as e:
                error_dialog = NotificationDialog("Error", f"An error occurred while importing PDF files: {e}")
                error_dialog.exec_()
                
    def handle_pdf_extract(self, file_path, output_dir, results):
        """Extract embedded images from PDF using PyPDF2."""
        import uuid
        
        try:
            # Create progress dialog
            progress = ImportFilesProgress("Extracting images from PDF...")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Extract images from PDF
            image_paths, total_pages = extract_images_from_pdf(file_path, output_dir)
            
            # Update progress dialog
            progress.setMaximum(len(image_paths))
            
            # Process each extracted image
            for i, image_path in enumerate(image_paths):
                file_id = uuid.uuid4()
                file_name = os.path.basename(image_path)
                
                data = build_ocr_data(file_id, image_path)
                results[file_id] = data
                
                # Update progress
                progress.setValue(i + 1)
                progress.setLabelText(f"Processing image {i + 1} of {len(image_paths)}...")
                
            progress.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to extract images from PDF: {e}")
                
    def convert_pdf_to_images(self, file_path, output_dir, results):
        """Convert PDF pages to images using pdf2image."""
        import uuid
        
        try:
            # Get PDF info
            poppler_path = self.get_poppler_path()
            
            logging.info("getting number of pages of PDF")
            pdf_info = pdfinfo_from_path(file_path, poppler_path=poppler_path)
            total_pages = pdf_info['Pages']
            logging.info(f"found {total_pages} pages")

            # Create progress dialog
            progress = ImportFilesProgress("Reading PDF file...", max_length=total_pages)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Process PDF in batches to avoid memory issues with large PDFs
            batch_size = 5
            for batch_start in range(1, total_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, total_pages)
                
                progress.setLabelText(f"Converting pages {batch_start} to {batch_end}...")
                
                # Convert batch of pages
                pages = convert_from_path(
                    file_path, 
                    dpi=300, 
                    first_page=batch_start, 
                    last_page=batch_end,
                    poppler_path=poppler_path
                )
                
                # Save each page
                for i, page in enumerate(pages):
                    page_num = batch_start + i
                    progress.setValue(page_num)
                    
                    # Save the page as an image
                    image_path = os.path.join(output_dir, f"page_{page_num}.png")
                    page.save(image_path, "PNG")
                    
                    # Create OCR data for this page
                    file_id = uuid.uuid4()
                    file_name = f"Page {page_num}"
                    
                    data = build_ocr_data(file_id, image_path)
                    results[file_id] = data
            
            progress.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import PDF: {e}")

    def import_files(self, results: Dict[UUID, OCRData]):
        self._dataview_model.add_data(results)

    def save(self):
        _ocr_data = self._dataview_model.get_data()
        ocred_lines = 0

        for k, data in _ocr_data.items():
            if data.ocr_lines is not None and len(data.ocr_lines) > 0:
                ocred_lines += 1

        if not len(_ocr_data) > 0:
            info_box = NotificationDialog(
                "Saving Annotations",
                "The current project contains no data."
            )
            info_box.exec()

        elif not ocred_lines > 0:
            info_box = NotificationDialog(
                "Saving Annotations",
                "The current contains no OCR data. Please run OCR first."
            )
            info_box.exec()

        else:
            dialog = ExportDialog(list(_ocr_data.values()), self._settingsview_model)
            dialog.setStyleSheet(DARK)
            dialog.exec()

    def select_page(self, index: int):
        self.image_gallery.select_page(index)

    def run_ocr(self, guid: UUID):
        data = self._dataview_model.get_data_by_guid(guid)

        if os.path.isfile(data.image_path):
            img = cv2.imread(data.image_path)
            ocr_settings = self._settingsview_model.get_ocr_settings()
            status, result = self.ocr_pipeline.run_ocr(
                img,
                k_factor=ocr_settings.k_factor,
                bbox_tolerance=ocr_settings.bbox_tolerance,
                merge_lines=ocr_settings.merge_lines,
                use_tps=ocr_settings.dewarping)

            if status == OpStatus.SUCCESS:
                mask, line_data, page_text, angle = result
                self._dataview_model.update_ocr_data(guid, page_text)
                self._dataview_model.update_page_data(guid, line_data, mask, angle)
            else:
                dialog = NotificationDialog("Failed Running OCR", f"Failed to run OCR on selected image.\n\n{result}")
                dialog.exec()
        else:
            dialog = NotificationDialog("Image not found", "The selected image could not be read from disk.")
            dialog.exec()

    def run_batch_ocr(self):
        _data = self._dataview_model.get_data()
        _data = list(_data.values())

        if _data is not None and len(_data) > 0:
            # Get currently selected OCR model from the pipeline
            current_model = None
            if self.ocr_pipeline is not None and self.ocr_pipeline.ocr_model_config is not None:
                # Find the model with matching config
                for model in self._settingsview_model.get_ocr_models():
                    if model.config == self.ocr_pipeline.ocr_model_config:
                        current_model = model
                        break
            
            batch_dialog = BatchOCRDialog(
                data=_data,
                ocr_pipeline=self.ocr_pipeline,
                ocr_models=self._settingsview_model.get_ocr_models(),
                ocr_settings=self._settingsview_model.get_ocr_settings(),
                threadpool=self.threadpool,
                current_model=current_model  # Pass the currently selected model
            )
            batch_dialog.sign_ocr_result.connect(self.update_ocr_result)

            batch_dialog.setStyleSheet(DARK)
            batch_dialog.exec()
        else:
            dialog = NotificationDialog("No data", "Project contains no data.")
            dialog.exec()

    def update_ocr_result(self, result: OCResult, silent: bool = False):
        if result is not None:
            self._dataview_model.update_ocr_data(result.guid, result.text, silent)
            self._dataview_model.update_page_data(result.guid, result.lines, result.mask, result.angle, silent)

        else:
            dialog = NotificationDialog("Failed Running OCR", "Failed to run OCR on selected image.")
            dialog.exec()

    def handle_ocr_batch_result(self, batch_result: Dict[UUID, OCResult]):
        if batch_result is not None:
            for _, result in batch_result.values():
                self.update_ocr_result(result, silent=True)
        else:
            dialog = NotificationDialog("Failed Running OCR", "Failed to run OCR on selected image.")
            dialog.exec()

    def handle_settings(self):
        _models = self._settingsview_model.get_ocr_models()
        _app_settings = self._settingsview_model.get_app_settings()
        _ocr_settings = self._settingsview_model.get_ocr_settings()

        dialog = SettingsDialog(_app_settings, _ocr_settings, _models)
        dialog.setStyleSheet(DARK)

        app_settings, ocr_settings, ocr_models = dialog.exec()
        self._settingsview_model.save_app_settings(app_settings)
        self._settingsview_model.save_ocr_settings(ocr_settings)
        self._settingsview_model.update_ocr_models(ocr_models)
        
        current_line_config = self._settingsview_model.get_line_model()

        if self.ocr_pipeline is not None:
            self.ocr_pipeline.update_line_detection(current_line_config)

    def update_ocr_model(self, ocr_model: OCRModel):
        if self.ocr_pipeline is not None:
            self.ocr_pipeline.update_ocr_model(ocr_model.config)
        else:
            line_model_config = self._settingsview_model.get_line_model()
            self.ocr_pipeline = OCRPipeline(self.platform, ocr_model.config, line_model_config)

    def get_poppler_path(self):
        # Return cached path if we've already found it
        if self._poppler_path is not None:
            return self._poppler_path
            
        try:
            # Determine base path depending on whether we're running from a bundled app or in development
            if getattr(sys, 'frozen', False):
                # Running in a bundled app
                base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
                print(f"Running from bundled app, base path: {base_path}")
            else:
                # Running in development mode
                base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                print(f"Running in development mode, base path: {base_path}")
                return None
            
            # Poppler is always in ./poppler/bin
            poppler_path = os.path.join(base_path, 'poppler', 'bin')
            print(f"Looking for Poppler at: {poppler_path}")
            
            # Check if pdfinfo exists in this path
            pdfinfo_path = os.path.join(poppler_path, 'pdfinfo')
            if platform.system() == 'Windows':
                pdfinfo_path += '.exe'
            
            print(f"Checking for pdfinfo at: {pdfinfo_path}")
            if os.path.exists(pdfinfo_path):
                print(f"Found Poppler at: {poppler_path}")
                
                # Cache the result
                self._poppler_path = poppler_path
                return poppler_path
            else:
                QMessageBox.critical(self, "Error", f"Poppler binaries not found at expected location: {poppler_path}")
                print(f"Poppler binaries not found at expected location: {poppler_path}")
                return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error finding Poppler: {e}")
            return None
