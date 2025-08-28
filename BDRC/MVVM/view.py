import cv2
import sys
import re
import platform
import os
from uuid import UUID
from typing import Dict, List
from PySide6.QtCore import Signal, Qt, QThreadPool, QThread
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QMessageBox, QFileDialog, QProgressDialog, QApplication, QToolTip
from PySide6.QtGui import QMovie, QClipboard
from pdf2image import convert_from_path, pdfinfo_from_path
from BDRC.Styles import DARK
from BDRC.Inference import OCRPipeline
from BDRC.Data import OpStatus, Platform, OCRData, OCRModel, OCResult
from BDRC.Utils import build_ocr_data, get_filename, create_dir
from BDRC.Widgets.Dialogs import NotificationDialog, ImportFilesProgress, PDFImportDialog, TextInputDialog, ExportDialog, SettingsDialog, BatchOCRDialog
from BDRC.utils.pdf_extract import extract_images_from_pdf
from BDRC.Widgets.Layout import HeaderTools, ImageGallery, Canvas, TextView
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
import logging

# Thread for asynchronous OCR
class _OCRThread(QThread):
    ocr_finished = Signal(object, object, object)  # status, result, guid
    def __init__(self, pipeline, img, settings, guid, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.img = img
        self.settings = settings
        self.guid = guid
    def run(self):
        status, result = self.pipeline.run_ocr(
            self.img,
            k_factor=self.settings.k_factor,
            bbox_tolerance=self.settings.bbox_tolerance,
            merge_lines=self.settings.merge_lines,
            use_tps=self.settings.dewarping
        )
        self.ocr_finished.emit(status, result, self.guid)

class MainView(QWidget):
    s_handle_import = Signal()
    s_handle_pdf_import = Signal()
    s_handle_page_select = Signal(int)
    s_on_file_save = Signal()
    s_run_ocr = Signal(UUID)
    s_run_batch_ocr = Signal()
    s_handle_settings = Signal()

    def __init__(self, data_view: DataViewModel, settings_view: SettingsViewModel, platform: Platform, translation_manager=None):
        super().__init__()
        self.setObjectName("MainView")
        self.setContentsMargins(0, 0, 0, 0)
        self._data_view = data_view
        self._settings_view = settings_view
        self.platform = platform
        self.translation_manager = translation_manager
        self.resource_dir = self._settings_view.get_execution_dir()
        self.default_font = self._settings_view.get_default_font_path()

        self.header_tools = HeaderTools(self._data_view, self._settings_view, translation_manager=self.translation_manager)
        self.canvas = Canvas(self.resource_dir )
        self.text_view = TextView(platform=self.platform, dataview=self._data_view, execution_dir=self.resource_dir, font_path=self.default_font, translation_manager=self.translation_manager)
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
        # enable save and copy when any OCR record updates
        self._data_view.s_record_changed.connect(lambda data: self.header_tools.toolbox.btn_save.setEnabled(True))
        self._data_view.s_record_changed.connect(lambda data: self.header_tools.toolbox.btn_copy_all.setEnabled(True))
        # Update main view when page switched via header
        self._data_view.s_data_auto_selected.connect(self.set_data)

        # connect to tool signals
        self.header_tools.toolbox.s_new.connect(self.handle_new)
        self.header_tools.toolbox.s_import_files.connect(self.handle_import)
        self.header_tools.toolbox.s_import_pdf.connect(self.handle_pdf_import)
        self.header_tools.toolbox.s_save.connect(self.handle_file_save)
        self.header_tools.toolbox.s_on_select_model.connect(self.handle_model_selection)
        self.header_tools.toolbox.s_run.connect(self.handle_run)
        self.header_tools.toolbox.s_run_all.connect(self.handle_batch_run)
        self.header_tools.toolbox.s_copy_all.connect(self.handle_copy_all)
        self.header_tools.toolbox.s_settings.connect(self.handle_settings)
        self.header_tools.page_switcher.s_on_page_changed.connect(self.handle_update_page)

        self.current_guid = None

    def handle_new(self):
        self._data_view.clear_data()

    def update_data(self, data: List[OCRData]):
        self.header_tools.update_page_count(len(data))
        # update button states based on data/presence of pages
        tb = self.header_tools.toolbox
        has_pages = len(data) > 0
        tb.btn_run.setEnabled(has_pages)
        tb.btn_run_all.setEnabled(has_pages)
        # disable save/export and copy until OCR text exists
        tb.btn_save.setEnabled(False)
        tb.btn_copy_all.setEnabled(False)

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

    def handle_copy_all(self):
        # collect OCR text from all pages
        data_dict = self._data_view.get_data()
        texts = []
        for data in data_dict.values():
            if data.ocr_lines:
                texts.append("\n".join([line.text for line in data.ocr_lines]))
        full_text = "\n\n".join(texts)
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text, QClipboard.Mode.Clipboard)
        # show popover near copy button
        btn = self.header_tools.toolbox.btn_copy_all
        pos = btn.mapToGlobal(btn.rect().bottomLeft())
        QToolTip.showText(pos, "Copied all text to clipboard", btn)

    def handle_settings(self):
        self.s_handle_settings.emit()

    def handle_update_page(self, index: int):
        self._data_view.select_data_by_index(index)

    def set_data(self, data: OCRData):
        self.canvas.set_preview(data)
        self.current_guid = data.guid

    def clear_data(self):
        self.canvas.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_overlay'):
            self._overlay.setGeometry(self.rect())

    def run_ocr(self, guid: UUID):
        data = self._dataview_model.get_data_by_guid(guid)
        if not os.path.isfile(data.image_path):
            NotificationDialog("Image not found", "The selected image could not be read from disk.").exec()
            return
        img = cv2.imread(data.image_path)
        ocr_settings = self._settings_view.get_ocr_settings()
        # show indeterminate progress dialog and lock UI
        self._progress_dialog = QProgressDialog("Running OCR...", None, 0, 0, self)
        self._progress_dialog.setWindowModality(Qt.ApplicationModal)
        self._progress_dialog.setCancelButton(None)
        self._progress_dialog.setWindowTitle("Please wait")
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.show()
        self.setEnabled(False)
        # start thread
        thread = _OCRThread(self.ocr_pipeline, img, ocr_settings, guid)
        self._ocr_thread = thread  # keep reference
        thread.ocr_finished.connect(self._on_thread_finished)
        thread.start()

    def _on_thread_finished(self, status, result, guid):
        # close progress & restore UI
        if hasattr(self, '_progress_dialog'):
            self._progress_dialog.close()
        self.setEnabled(True)
        # handle OCR result
        if status == OpStatus.SUCCESS:
            mask, line_data, page_text, angle = result
            self._dataview_model.update_ocr_data(guid, page_text)
            self._dataview_model.update_page_data(guid, line_data, mask, angle)
        else:
            NotificationDialog("Failed Running OCR", f"Failed to run OCR on selected image.\n\n{result}").exec()


class AppView(QWidget):
    def __init__(self,
                 dataview_model: DataViewModel,
                 settingsview_model: SettingsViewModel,
                 platform: Platform,
                 translation_manager=None):
        super().__init__()

        self.setObjectName("MainWindow")
        self.setWindowTitle("BDRC OCR [BETA] 0.3")
        self.setContentsMargins(0, 0, 0, 0)
        self.platform = platform
        self.threadpool = QThreadPool()
        self._dataview_model = dataview_model
        self._settingsview_model = settingsview_model
        self.translation_manager = translation_manager

        self.tmp_dir = self._settingsview_model.get_tmp_dir()
        self.resource_dir = self._settingsview_model.get_execution_dir()

        self.image_gallery = ImageGallery(self._dataview_model, self.threadpool, self.resource_dir)
        self.main_container = MainView(self._dataview_model, self._settingsview_model, self.platform, self.translation_manager)
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
                                # Extract embedded images using pypdf
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
                            # Extract embedded images using pypdf
                            self.handle_pdf_extract(file_path, pdf_dir, results)
                        else:
                            # Convert pages to images using pdf2image
                            self.convert_pdf_to_images(file_path, pdf_dir, results)
                
                if results:
                    self.import_files(results)
                else:
                    NotificationDialog("No images found", "No images could be extracted from the selected PDF.").exec()
                    
            except Exception as e:
                error_dialog = NotificationDialog("Error", f"An error occurred while importing PDF files: {e}")
                error_dialog.exec_()
                
    def handle_pdf_extract(self, file_path, output_dir, results):
        """Extract embedded images from PDF using pypdf."""
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
            
            file_n = os.path.splitext(os.path.basename(file_path))[0]
            
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
                    image_path = os.path.join(output_dir, f"{file_n} - page {page_num}.png")
                    page.save(image_path, "PNG")
                    
                    # Create OCR data for this page
                    file_id = uuid.uuid4()
                    
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
            _ocr_settings = self._settingsview_model.get_ocr_settings()
            dialog = ExportDialog(list(_ocr_data.values()), _ocr_settings.output_encoding)
            dialog.setStyleSheet(DARK)
            dialog.exec()

    def select_page(self, index: int):
        self.image_gallery.select_page(index)

    def run_ocr(self, guid: UUID):
        data = self._dataview_model.get_data_by_guid(guid)
        if not os.path.isfile(data.image_path):
            NotificationDialog("Image not found", "The selected image could not be read from disk.").exec()
            return
        img = cv2.imread(data.image_path)
        ocr_settings = self._settingsview_model.get_ocr_settings()
        # show indeterminate progress dialog and lock UI
        self._progress_dialog = QProgressDialog("Running OCR...", None, 0, 0, self)
        self._progress_dialog.setWindowModality(Qt.ApplicationModal)
        self._progress_dialog.setCancelButton(None)
        self._progress_dialog.setWindowTitle("Please wait")
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.show()
        self.setEnabled(False)
        # run in background thread
        thread = _OCRThread(self.ocr_pipeline, img, ocr_settings, guid)
        self._ocr_thread = thread  # keep reference
        thread.ocr_finished.connect(self._on_ocr_finished)
        thread.start()

    def _on_ocr_finished(self, status, result, guid):
        # restore UI
        self.setEnabled(True)
        # close progress dialog
        if hasattr(self, '_progress_dialog'):
            self._progress_dialog.close()
        if status == OpStatus.SUCCESS:
            mask, line_data, page_text, angle = result
            self._dataview_model.update_ocr_data(guid, page_text)
            self._dataview_model.update_page_data(guid, line_data, mask, angle)
        else:
            NotificationDialog("Failed Running OCR", f"Failed to run OCR on selected image.\n\n{result}").exec()

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
        # Return cached path
        if self._poppler_path:
            return self._poppler_path

        try:
            # Collect base dirs for onefile and dev modes
            bases = []
            # Nuitka onefile temp extraction
            tmp = os.environ.get('NUITKA_ONEFILE_TEMP')
            if tmp:
                bases.append(tmp)
            # PyInstaller extraction
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                bases.append(sys._MEIPASS)
            # Executable directory
            exe_dir = os.path.dirname(sys.executable)
            bases.append(exe_dir)
            # CWD
            bases.append(os.getcwd())
            # Dev project root
            bases.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

            # Find pdfinfo in any base
            for base in bases:
                pdfinfo = os.path.join(base, 'poppler', 'bin', 'pdfinfo')
                if platform.system() == 'Windows':
                    pdfinfo += '.exe'
                if os.path.exists(pdfinfo):
                    poppler_bin = os.path.join(base, 'poppler', 'bin')
                    self._poppler_path = poppler_bin
                    print(f"Found Poppler at: {poppler_bin}")
                    # Update PATH and DYLD_LIBRARY_PATH
                    os.environ['PATH'] = poppler_bin + os.pathsep + os.environ.get('PATH', '')
                    if platform.system() == 'Darwin':
                        lib_dir = os.path.join(base, 'poppler', 'lib')
                        os.environ['DYLD_LIBRARY_PATH'] = lib_dir + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', '')
                    return poppler_bin

            # Not found at expected paths; try fallback recursive search under bases
            target = 'pdfinfo.exe' if platform.system() == 'Windows' else 'pdfinfo'
            for base in bases:
                for root, dirs, files in os.walk(base):
                    if target in files:
                        poppler_bin = root
                        self._poppler_path = poppler_bin
                        print(f"Found Poppler via fallback at: {poppler_bin}")
                        os.environ['PATH'] = poppler_bin + os.pathsep + os.environ.get('PATH', '')
                        if platform.system() == 'Darwin':
                            lib_dir = os.path.join(os.path.dirname(poppler_bin), 'lib')
                            os.environ['DYLD_LIBRARY_PATH'] = lib_dir + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', '')
                        return poppler_bin
            # Final failure: show attempted base directories
            checked = '\n'.join(bases)
            QMessageBox.critical(self, 'Error', f"Poppler binaries not found. Checked:\n{checked}")
            return None
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error finding Poppler: {e}")
            return None
