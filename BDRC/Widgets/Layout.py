import os
from uuid import UUID
from typing import List
from PySide6.QtCore import Qt
from BDRC.Data import Encoding, OCRLine, OCRLineUpdate, Platform
from BDRC.Utils import get_filename
from BDRC.Data import OCRData, OCRModel
from BDRC.Widgets.GraphicItems import ImagePreview
from BDRC.Widgets.Buttons import MenuButton, TextToolsButton
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from BDRC.Widgets.Dialogs import TextInputDialog
from BDRC.Translation import tr

from PySide6.QtCore import Signal, QPoint, QPointF, QSize, QEvent, QRectF, QThreadPool
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPen,
    QImage,
    QPixmap,
    QPainter,
    QPainterPath,
    QResizeEvent,
    QFontDatabase
)

from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QScrollBar,
    QWidget,
    QLabel,
    QSpacerItem,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QFrame,
    QListView,
    QToolTip
)


class HeaderTools(QFrame):
    def __init__(self, data_view: DataViewModel, settings_view: SettingsViewModel, icon_size: int = 48, translation_manager=None):
        super().__init__()
        self.setObjectName("HeaderTools")
        self.data_view = data_view
        self.settings_view = settings_view
        self.translation_manager = translation_manager
        self.execution_dir = self.settings_view.get_execution_dir()
        self.toolbox = ToolBox(self.execution_dir, ocr_models=self.settings_view.get_ocr_models(), icon_size=icon_size, translation_manager=self.translation_manager)
        self.page_switcher = PageSwitcher(self.execution_dir, icon_size=icon_size, translation_manager=self.translation_manager)

        # bind signals
        self.data_view.s_data_selected.connect(self.set_page_index)
        self.settings_view.s_ocr_models_changed.connect(self.update_ocr_models)

        # build layout
        self.spacer = QLabel()
        self.spacer.setFixedWidth(30)
        self.end_spacer = QLabel()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.toolbox)
        self.layout.addWidget(self.spacer)
        self.layout.addWidget(self.page_switcher)
        self.layout.addWidget(self.end_spacer)

        self.layout.setContentsMargins(0, 4, 4, 4)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(self.layout)

    def update_page_count(self, amount: int):
        self.page_switcher.max_pages = amount

    def set_page_index(self, data: OCRData):
        _index = self.data_view.get_data_index(data.guid)
        self.page_switcher.update_page(_index)

    def update_ocr_models(self):
        _ocr_models = self.settings_view.get_ocr_models()
        self.toolbox.update_ocr_models(_ocr_models)


class ToolBox(QWidget):
    s_new = Signal()
    s_import_files = Signal()
    s_import_pdf = Signal()
    s_save = Signal()
    s_run = Signal()
    s_run_all = Signal()
    s_copy_all = Signal()
    s_settings = Signal()
    s_language_switch = Signal()
    s_update_page = Signal(int)
    s_on_select_model = Signal(OCRModel)

    def __init__(self, execution_dir: str, ocr_models: List[OCRModel] | None, icon_size: int = 64, translation_manager=None):
        super().__init__()
        self.setObjectName("ToolBox")
        self.ocr_models = ocr_models
        self.icon_size = icon_size
        self.translation_manager = translation_manager
        self.setFixedHeight(self.icon_size+18)
        self.setMinimumWidth(720)

        self.new_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "new_light.png")
        self.import_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "import.png")
        self.import_pdf_icon = os.path.join(execution_dir, "Assets", "Textures", "pdf_import.png")
        self.copy_all_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "copy.png")
        self.save_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "save-disc.png")
        self.run_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "play_btn.png")
        self.run_all_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "play_all_btn.png")
        self.settings_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "settings.png")
        self.language_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "settings.png")  # Reuse settings icon for now

        self.btn_new = MenuButton(
            tr("New Project"),
            self.new_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_import_images = MenuButton(
            tr("Import Images"),
            self.import_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_import_pdf = MenuButton(
            tr("Import PDF"),
            self.import_pdf_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_copy_all = MenuButton(
            tr("Copy All"),
            self.copy_all_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )
        self.btn_copy_all.setEnabled(False)

        self.btn_save = MenuButton(
            tr("Save Output"),
            self.save_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run = MenuButton(
            tr("Run OCR"),
            self.run_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run_all = MenuButton(
            tr("Run OCR on all images"),
            self.run_all_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_settings = MenuButton(
            tr("Settings"),
            self.settings_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        # Language switcher button
        self.btn_language = MenuButton(
            tr("Switch Language"),
            self.language_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        # model selection
        self.model_selection = QComboBox()
        self.model_selection.setFixedHeight(int(self.icon_size * 0.75))
        self.model_selection.setObjectName("ModelSelection")
        self.model_selection.setCursor(Qt.CursorShape.PointingHandCursor)

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)
            # Restore persisted model selection from QSettings
            from PySide6.QtCore import QSettings
            settings = QSettings("BDRC", "TibetanOCRApp")
            persisted_model_name = settings.value("main/model_name", None)
            restored = False
            self.model_selection.blockSignals(True)
            if persisted_model_name is not None:
                for i, model in enumerate(self.ocr_models):
                    if model.name == persisted_model_name:
                        self.model_selection.setCurrentIndex(i)
                        restored = True
                        break
            if not restored:
                self.model_selection.setCurrentIndex(0)
            self.model_selection.blockSignals(False)

        # self.model_selection.activated.connect(self.on_select_ocr_model)
        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        # connect button signals
        self.btn_new.clicked.connect(self.new)
        self.btn_import_images.clicked.connect(self.load_images)
        self.btn_import_pdf.clicked.connect(self.import_pdf)
        self.btn_copy_all.clicked.connect(self.copy_all)
        self.btn_save.clicked.connect(self.save)
        self.btn_run.clicked.connect(self.run)
        self.btn_run_all.clicked.connect(self.run_all)
        self.btn_settings.clicked.connect(self.settings)
        self.btn_language.clicked.connect(self.switch_language)

        # disable buttons until data or OCR ready
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_copy_all.setEnabled(False)

        # build layout
        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.layout.setContentsMargins(10, 0, 0, 0)

        self.layout.addWidget(self.btn_new)
        self.layout.addWidget(self.btn_import_images)
        self.layout.addWidget(self.btn_import_pdf)
        self.layout.addWidget(self.btn_copy_all)
        self.layout.addWidget(self.btn_save)
        self.layout.addWidget(self.btn_run)
        self.layout.addWidget(self.btn_run_all)
        self.layout.addWidget(self.btn_settings)
        self.layout.addWidget(self.btn_language)
        self.layout.addWidget(self.model_selection)
        self.setLayout(self.layout)

        # Emit initial model selection after setup
        if self.ocr_models and self.model_selection.currentIndex() >= 0:
            self.on_select_ocr_model(self.model_selection.currentIndex())

    def new(self):
        self.s_new.emit()

    def load_images(self):
        self.s_import_files.emit()

    def import_pdf(self):
        self.s_import_pdf.emit()

    def copy_all(self):
        self.s_copy_all.emit()

    def save(self):
        self.s_save.emit()

    def run(self):
        self.s_run.emit()

    def run_all(self):
        self.s_run_all.emit()

    def settings(self):
        self.s_settings.emit()

    def switch_language(self):
        """Handle language switching."""
        if self.translation_manager:
            current_lang = self.translation_manager.get_current_language()
            # Toggle between English and Tibetan
            new_lang = "bo" if current_lang == "en" else "en"
            success = self.translation_manager.switch_language(new_lang)
            
            if success:
                # Save the new language preference
                from PySide6.QtCore import QSettings
                settings = QSettings("BDRC", "TibetanOCRApp")
                settings.setValue("ui/language", new_lang)
                
                # Update button texts
                self.update_button_texts()
        else:
            self.s_language_switch.emit()

    def update_button_texts(self):
        """Update button texts after language change."""
        self.btn_new.setToolTip(tr("New Project"))
        self.btn_import_images.setToolTip(tr("Import Images"))
        self.btn_import_pdf.setToolTip(tr("Import PDF"))
        self.btn_copy_all.setToolTip(tr("Copy All"))
        self.btn_save.setToolTip(tr("Save Output"))
        self.btn_run.setToolTip(tr("Run OCR"))
        self.btn_run_all.setToolTip(tr("Run OCR on all images"))
        self.btn_settings.setToolTip(tr("Settings"))
        self.btn_language.setToolTip(tr("Switch Language"))

    def update_page(self, index: int):
        self.s_update_page.emit(index)

    def on_select_ocr_model(self, index: int):
        # Persist the selected model name
        from PySide6.QtCore import QSettings
        settings = QSettings("BDRC", "TibetanOCRApp")
        if self.ocr_models and 0 <= index < len(self.ocr_models):
            settings.setValue("main/model_name", self.ocr_models[index].name)
        self.s_on_select_model.emit(self.ocr_models[index])

    def update_ocr_models(self, ocr_models: List[OCRModel]):
        # Remember currently selected model name before clearing
        current_model_name = None
        if self.model_selection.currentIndex() >= 0 and self.ocr_models is not None:
            current_model_name = self.ocr_models[self.model_selection.currentIndex()].name

        self.ocr_models = ocr_models

        from PySide6.QtCore import QSettings
        settings = QSettings("BDRC", "TibetanOCRApp")
        persisted_model_name = settings.value("main/model_name", None)

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            # Block signals temporarily to avoid triggering model reload when restoring selection
            self.model_selection.blockSignals(True)
            self.model_selection.clear()

            for model in self.ocr_models:
                self.model_selection.addItem(model.name)
            
            # Priority: 1. persisted model, 2. previously selected model
            restored = False
            if persisted_model_name is not None:
                for i, model in enumerate(self.ocr_models):
                    if model.name == persisted_model_name:
                        self.model_selection.setCurrentIndex(i)
                        restored = True
                        break
            if not restored and current_model_name is not None:
                for i, model in enumerate(self.ocr_models):
                    if model.name == current_model_name:
                        self.model_selection.setCurrentIndex(i)
                        break
            
            # Re-enable signals after we've set everything up
            self.model_selection.blockSignals(False)


class PageSwitcher(QFrame):
    s_on_page_changed = Signal(int)

    def __init__(self, execution_dir: str, pages: int = 0, icon_size: int = 36, translation_manager=None):
        super().__init__()
        self.setObjectName("PageSwitcher")
        self.icon_size = icon_size
        self.max_pages = pages
        self.current_index = 0
        self.translation_manager = translation_manager

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.current_page = QLabel("")
        self.current_page.setObjectName("PageNumberLabel")
        self.current_page.setFixedWidth(100)
        self.current_page.setFixedHeight(icon_size)
        self.current_page.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prev_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "prev.png")
        self.next_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "next.png")

        self.prev_button = MenuButton(
            tr("Previous image"),
            self.prev_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.next_button = MenuButton(
            tr("Next image"),
            self.next_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        # build layout
        self.layout.addWidget(self.prev_button)
        self.layout.addWidget(self.current_page)
        self.layout.addWidget(self.next_button)
        self.setLayout(self.layout)

        self.prev_button.clicked.connect(self.prev)
        self.next_button.clicked.connect(self.next)

    def update_page(self, index: int):
        self.current_index = index
        self.current_page.setText(str(self.current_index+1))

    def prev(self):
        next_index = self.current_index - 1

        if next_index >= 0:
            self.update_page(next_index)
            self.s_on_page_changed.emit(next_index)

    def next(self):
        next_index = self.current_index + 1

        if not next_index > self.max_pages-1:
            self.update_page(next_index)
            self.s_on_page_changed.emit(next_index)


class PTGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(parent)
        self.setObjectName("PTGraphicsView")
        self.scene = scene
        self.setScene(self.scene)
        self.setMinimumHeight(200)

        self.zoom_in_factor = 0.8
        self.zoom_clamp = False
        self.default_zoom_step = 10
        self.current_zoom_step = 10
        self.zoom_range = [0, 20]

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.default_scrollbar_policy = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        self.setHorizontalScrollBarPolicy(self.default_scrollbar_policy)
        self.setVerticalScrollBarPolicy(self.default_scrollbar_policy)

        self.verticalScrollBar().setSliderPosition(1)
        self.horizontalScrollBar().setSliderPosition(1)

        self.v_scrollbar = QScrollBar(self)
        self.v_scrollbar.setObjectName("VerticalScrollBar")
        self.h_scrollbar = QScrollBar(self)
        self.v_scrollbar.setObjectName("HorizontalScrollbar")

        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)

        self.v_scrollbar.setStyleSheet(
            """

            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }

            QScrollBar::handle:vertical {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #C80021;
            }

            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-line:vertical {
                height: 0px;
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """
        )

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
        """
        )


    def enable_rubberband(self):
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def disable_rubberband(self):
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def hide_scrollbars(self):
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def show_scrollbars(self):
        self.setHorizontalScrollBarPolicy(self.default_scrollbar_policy)
        self.setVerticalScrollBarPolicy(self.default_scrollbar_policy)

    def wheelEvent(self, event):
        if event.source() == Qt.MouseEventSource.MouseEventSynthesizedBySystem:
            self.handle_touch_zoom(event.angleDelta().y())
        else:
            self.handle_mouse_zoom(event.angleDelta().y())

    def handle_touch_zoom(self, y_delta: int):
        if y_delta > 6:
            if self.zoom_range[0] <= self.current_zoom_step < self.zoom_range[-1]:
                zoom_factor = 0.99
                self.current_zoom_step += 0.1

                if self.current_zoom_step > self.zoom_range[-1]:
                    self.current_zoom_step = self.zoom_range[-1]
                    return
                self.scale(zoom_factor, zoom_factor)

        elif y_delta < - 6:
            if self.zoom_range[0] < self.current_zoom_step <= self.zoom_range[-1]:
                zoom_factor = 1.01
                self.current_zoom_step -= 0.1
                if self.current_zoom_step < self.zoom_range[0]:
                    self.current_zoom_step = self.zoom_range[0]
                    return
                self.scale(zoom_factor, zoom_factor)

    def handle_mouse_zoom(self, y_delta: int):
        if y_delta > 0:
            if self.zoom_range[0] <= self.current_zoom_step < self.zoom_range[-1]:
                zoom_factor = self.zoom_in_factor
                self.current_zoom_step += 1
                self.scale(zoom_factor, zoom_factor)

        else:
            if self.zoom_range[0] < self.current_zoom_step <= self.zoom_range[-1]:
                zoom_factor = 1 / self.zoom_in_factor
                self.current_zoom_step -= 1
                self.scale(zoom_factor, zoom_factor)

    def reset_scaling(self):
        self.resetTransform()
        self.current_zoom_step = self.default_zoom_step

    def resizeEvent(self, event):
        for item in self.scene.items():
            if isinstance(item, ImagePreview):
                b_rect = item.boundingRect()
                self.fit_in_view(b_rect)
        return super().resizeEvent(event)
    
    def fit_in_view(self, brect: QRectF):
        if not self.default_zoom_step < self.current_zoom_step < self.default_zoom_step:
            _target_zoom_step = self.default_zoom_step - self.current_zoom_step

        else:
            return

        _zoom_factor = self.zoom_in_factor ** _target_zoom_step
        self.scale(_zoom_factor, _zoom_factor)
        self.current_zoom_step = 10


class PTGraphicsScene(QGraphicsScene):
    # see: https://forum.qt.io/topic/101616/pyside2-qtcore-signal-object-has-no-attribute-emit/4
    s_left_click_signal = Signal(QPoint)
    s_left_release_signal = Signal(QPoint)
    s_right_click_signal = Signal(QPoint)
    s_right_release_signal = Signal(QPoint)
    s_clear_selections = Signal()

    def __init__(self, execution_dir: str, scene, width: int = 2000, height: int = 2000, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._scene_width = width
        self._scene_height = height
        self._line_color = QColor("#2f2f2f")
        self._background_color = QColor("#393939")
        self._pen_light = QPen(self._line_color)
        self._pen_light.setWidth(10)
        self._bg_image = QPixmap(os.path.join(execution_dir, "Assets", "Textures", "background_grid.jpg"))

        self.set_scene(self._scene_width, self._scene_height)
        self.setBackgroundBrush(self._bg_image)

    def set_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0, 0, width, height)

    def add_item(self, item: QGraphicsItem, z_order: int):
        _brect = item.boundingRect()
        x = _brect.left()
        y = _brect.top()
        width = _brect.width()
        height = _brect.height()

        self.setSceneRect(x, y, width, height)
        item.setZValue(z_order)
        self.addItem(item)

    def remove_item(self, item: QGraphicsItem):
        self.removeItem(item)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(e)

    def get_current_item_pos(self) -> QPointF:
        canvas_items = self.items(order=Qt.SortOrder.AscendingOrder)

        for _item in canvas_items:
            if isinstance(_item, ImagePreview):
                return _item.scenePos()
            else:
                return QPointF(0, 0)

        return QPointF(0, 0)


class Canvas(QFrame):
    def __init__(self, execution_dir: str, width: int = 2000, height: int = 2000):
        super().__init__()

        self.default_width = width
        self.default_height = height
        self.setMinimumHeight(200)
        self.setObjectName("MainCanvas")

        self.current_width = self.default_width
        self.current_height = self.default_height
        self.current_item_pos = QPointF(0.0, 0.0)

        self.gr_scene = PTGraphicsScene(execution_dir, self, width=self.current_width, height=self.current_height)
        self.view = PTGraphicsView(self.gr_scene)
        self.view.setScene(self.gr_scene)

        self.toggle_prev_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "toggle_prev.png")
        self.fit_view_icon = os.path.join(execution_dir, "Assets", "Textures", "fit_to_canvas.png")
        self.zoom_in_icon = os.path.join(execution_dir, "Assets", "Textures", "plus_sign.png")
        self.zoom_out_icon = os.path.join(execution_dir, "Assets", "Textures", "minus_sign.png")

        self.toggle_prev_btn = MenuButton(
            "Toggle line preview",
            self.toggle_prev_btn_icon,
            width=26,
            height=26
        )
        self.toggle_prev_btn.setObjectName("CanvasToolButton")

        self.fit_in_btn = MenuButton(
            "Fit image in view",
            self.fit_view_icon,
            width=26,
            height=26
        )
        self.fit_in_btn.setObjectName("CanvasToolButton")

        self.zoom_in_btn = MenuButton(
            "Zoom in",
            self.zoom_in_icon,
            width=26,
            height=26
        )
        self.zoom_in_btn.setObjectName("CanvasToolButton")

        self.zoom_out_btn = MenuButton(
            "Zoom out",
            self.zoom_out_icon,
            width=26,
            height=26
        )
        self.zoom_out_btn.setObjectName("CanvasToolButton")

        # bind signals
        self.toggle_prev_btn.clicked.connect(self.handle_preview_toggle)
        self.fit_in_btn.clicked.connect(self.fit_in_view)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        self.canvas_tools_layout = QHBoxLayout()
        self.canvas_tools_layout.setContentsMargins(0, 0, 0, 6)
        self.canvas_tools_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.canvas_tools_layout.addWidget(self.toggle_prev_btn)
        self.canvas_tools_layout.addWidget(self.fit_in_btn)
        self.canvas_tools_layout.addWidget(self.zoom_in_btn)
        self.canvas_tools_layout.addWidget(self.zoom_out_btn)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.canvas_tools_layout)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

    def update_display_position(self, position: QPointF):
        self.current_item_pos = position

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            _new_size = event.size()
            self.current_width = _new_size.width()
            self.current_height = _new_size.height()
            self.gr_scene.setSceneRect(
                QRectF(0, 0, self.current_width, self.current_height)
            )

    def set_preview(self, data: OCRData):
        self.view.reset_scaling()
        self.gr_scene.clear()

        preview_item = ImagePreview(data.image_path, data.lines, data.angle)
        b_rect = preview_item.boundingRect()
        _pos = QPointF(0, 0)
        preview_item.setPos(_pos)

        self.gr_scene.add_item(preview_item, 1)
        self.view.fitInView(b_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def handle_preview_toggle(self):
        for item in self.gr_scene.items():
            if isinstance(item, ImagePreview):
                if item.is_in_preview:
                    item.show_image()
                else:
                    item.show_preview()

    def fit_in_view(self):
        #print("Canvas -> fit_in_view")
        scene_rect = self.gr_scene.sceneRect()
        #print(f"Canvas -> SceneRect: {scene_rect}")
        view_height = self.view.height()
        view_width = self.view.width()
        #print(f"Canvas -> ViewSize: {view_width}, {view_height}")

        for item in self.gr_scene.items():
            if isinstance(item, ImagePreview):
                b_rect = item.boundingRect()
                item.setPos(0, 0)
                self.view.fit_in_view(b_rect)


    def zoom_in(self):
        self.view.handle_mouse_zoom(-1)

    def zoom_out(self):
        self.view.handle_mouse_zoom(1)

    def clear(self):
        self.view.reset_scaling()
        self.gr_scene.clear()


class ImageList(QListWidget):
    s_on_selected_item = Signal(UUID)
    """
    https://stackoverflow.com/questions/64576846/how-to-paint-an-outline-when-hovering-over-a-qlistwidget-item
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setObjectName("ImageGalleryList")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)

        self.v_scrollbar = QScrollBar(self)
        self.h_scrollbar = QScrollBar(self)
        self.v_scrollbar.setStyleSheet("""
                                                             
            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }

            QScrollBar::handle:vertical {	
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{	
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {	
                background-color: #C80021;
            }
            
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
           
            QScrollBar::add-line:vertical {
                height: 0px;
            }
            
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """)

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
            """
        )

        self.setAutoScrollMargin(20)
        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(item)  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, ImageListWidget):
            self.s_on_selected_item.emit(_list_item_widget.guid)


class ImageThumb(QFrame):
    def __init__(self, q_image: QImage, width: int = 140, height: int = 80):
        super().__init__()
        self.target_width = width
        self.target_height = height
        self.max_height = 140
        self.round_rect_margin = 6
        self.round_rect_radius = 14
        self.current_width = self.target_width - 2 * self.round_rect_margin

        # Apply sharpening to improve clarity
        self.qimage = q_image.scaledToHeight(self.max_height, Qt.TransformationMode.SmoothTransformation)
        self.pixmap = QPixmap.fromImage(self.qimage)
        self.brush = QBrush(self.pixmap)

        self._pen_hover = QPen(QColor("#fce08d"))
        self._pen_hover.setWidth(6)

        self._pen_select = QPen(QColor("#ffad00"))
        self._pen_select.setWidth(6)

        self.source_img = QImage(self.current_width, self.max_height, QImage.Format.Format_ARGB32)
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(self.current_width, self.max_height, QImage.Format.Format_ARGB32)
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()
        self.clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin,
                self.round_rect_margin,
                -self.round_rect_margin,
                -self.round_rect_margin,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.hover_clip_path = QPainterPath()
        self.hover_clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin - 2,
                self.round_rect_margin - 2,
                -self.round_rect_margin + 2,
                -self.round_rect_margin + 2,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.is_hovered = False
        self.is_selected = False

    def resize_thumb(self, new_width: int):
        self.current_width = new_width
        self.source_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()

        self.clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin,
                self.round_rect_margin,
                -self.round_rect_margin,
                -self.round_rect_margin,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.hover_clip_path = QPainterPath()
        self.hover_clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin - 2,
                self.round_rect_margin - 2,
                -self.round_rect_margin + 2,
                -self.round_rect_margin + 2,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setClipPath(self.clip_path)
        painter.drawImage(0, 0, self.qimage)

        if self.is_hovered:
            painter.setClipPath(self.hover_clip_path)
            painter.drawImage(0, 0, self.qimage)
            path_outline = QPainterPath()
            path_outline.addRoundedRect(
                self.round_rect_margin,
                self.round_rect_margin,
                self.current_width - 2 * self.round_rect_margin,
                self.max_height - 2 * self.round_rect_margin,
                self.round_rect_radius,
                self.round_rect_radius,
            )

            painter.setPen(self._pen_hover)
            painter.drawPath(path_outline.simplified())

        elif self.is_selected:
            painter.setClipPath(self.hover_clip_path)
            painter.drawImage(0, 0, self.qimage)
            path_outline = QPainterPath()
            path_outline.addRoundedRect(
                self.round_rect_margin,
                self.round_rect_margin,
                self.current_width - 2 * self.round_rect_margin,
                self.max_height - 2 * self.round_rect_margin,
                self.round_rect_radius,
                self.round_rect_radius,
            )

            painter.setPen(self._pen_select)
            painter.drawPath(path_outline.simplified())

        else:
            painter.setClipPath(self.clip_path)
            painter.drawImage(0, 0, self.qimage)

        painter.end()


class ImageListWidget(QWidget):
    s_delete_image = Signal(UUID)

    def __init__(self, guid: UUID, image_path: str, q_image: QImage, width: int, height: int, execution_dir: str):
        super().__init__()
        self.guid = guid
        self.image_path = image_path
        self.file_name = get_filename(image_path)
        self.base_width = width
        self.base_height = height

        self.thumb = ImageThumb(q_image, width=self.base_width)
        self.label = QLabel()
        self.label.setObjectName("DefaultLabel")
        self.label.setContentsMargins(0, 0, 0, 0)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setText(self.file_name)

        self.btn_delete_icon = os.path.join(execution_dir, "Assets", "Textures", "delete_icon.png")
        self.icon_size = 24
        self.btn_delete = MenuButton(
            "Delete image",
            self.btn_delete_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.setBaseSize(self.base_width, self.base_height)

        self.v_layout = QVBoxLayout()
        self.v_layout.setSpacing(0)
        self.v_layout.addWidget(self.thumb)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.label)
        self.h_layout.addWidget(self.btn_delete)
        self.v_layout.addLayout(self.h_layout)

        self.setLayout(self.v_layout)
        self.is_active = False
        
        # bind delete signal
        self.btn_delete.clicked.connect(self.delete_image)

    def delete_image(self):
        self.s_delete_image.emit(self.guid)

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            self.thumb.resize_thumb(event.size().width()-20)

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.thumb.is_hovered = True
            self.thumb.is_selected = False
            self.thumb.update()

        elif event.type() == QEvent.Type.Leave:
            self.thumb.is_hovered = False

            if self.is_active:
                self.thumb.is_selected = True
            else:
                self.thumb.is_selected = False

            self.thumb.update()

        return super().event(event)

    def select(self):
        self.is_active = True
        self.thumb.is_selected = True
        self.thumb.is_hovered = False
        self.thumb.update()

    def unselect(self):
        self.is_active = False
        self.thumb.is_selected = False
        self.thumb.is_hovered = False
        self.thumb.update()


class ImageGallery(QFrame):
    def __init__(self, viewmodel: DataViewModel, pool: QThreadPool, execution_dir: str):
        super().__init__()
        self.view_model = viewmodel
        self.pool = pool
        self.setObjectName("ImageGallery")
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumHeight(600)
        self.setMinimumWidth(250)
        self.setMaximumWidth(500)
        self.import_dialog = None
        self.execution_dir = execution_dir

        # build layout
        self.image_label = QLabel(self)
        self.image_label.setContentsMargins(6, 0, 0, 0)
        self.image_pixmap = QPixmap(os.path.join(execution_dir, "Assets", "Textures", "BDRC_Logo.png")).scaled(
            QSize(140, 90),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(self.image_pixmap)

        self.layout = QVBoxLayout()
        self.spacer = QSpacerItem(180, 10)
        self.image_list = ImageList(self)

        self.layout.addWidget(self.image_label)
        self.layout.addItem(self.spacer)
        self.layout.addWidget(self.image_list)
        self.setLayout(self.layout)

        # connect signals
        self.view_model.s_data_changed.connect(self.add_data)
        self.view_model.s_data_size_changed.connect(self.refresh_data)
        self.view_model.s_data_cleared.connect(self.clear_data)
        self.view_model.s_data_auto_selected.connect(self.focus_page)
        self.image_list.s_on_selected_item.connect(self.handle_item_selection)

        self.current_size = self.sizeHint()
        self.current_width = self.current_size.width()
        self.current_height = self.current_size.height()
        self.image_list.resize(self.current_width, self.current_height)
        

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            _new_size = event.size()
            self.current_width = _new_size.width()
            #self.image_list.resizeContents(self.current_width)

    def handle_item_selection(self, guid: UUID):
        for idx in range(self.image_list.count()):
                item = self.image_list.item(idx)
                item_widget = self.image_list.itemWidget(item)

                if isinstance(item_widget, ImageListWidget):
                    if item_widget.guid == guid:
                        item_widget.select()
                    else:
                        item_widget.unselect()

        self.view_model.select_data_by_guid(guid)

    def select_page(self, index: int):
        for idx in range(self.image_list.count()):
            if idx == index:
                item = self.image_list.item(idx)
                item.setSelected(True)
                item_widget = self.image_list.itemWidget(item)

                if isinstance(item_widget, ImageListWidget):
                    item_widget.select()
                    self.view_model.select_data_by_guid(item_widget.guid)
            else:
                item = self.image_list.item(idx)
                item.setSelected(False)
                item_widget = self.image_list.itemWidget(item)

                if isinstance(item_widget, ImageListWidget):
                    item_widget.unselect()

    def focus_page(self, data: OCRData):
        for idx in range(self.image_list.count()):
            item = self.image_list.item(idx)
            item_widget = self.image_list.itemWidget(item)

            if isinstance(item_widget, ImageListWidget):
                if item_widget.guid == data.guid:
                    item.setSelected(True)  # what does this built-in method actually do..?
                    item_widget.select()
                    self.image_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                else:
                    item.setSelected(False)
                    item_widget.unselect()

    def add_image_widget(self, data: OCRData, target_width: int):
        image_item = QListWidgetItem()
        image_item.setSizeHint(QSize(target_width, 200))
        image_widget = ImageListWidget(
            data.guid,
            data.image_path,
            data.qimage,
            width=target_width,
            height=200,
            execution_dir= self.execution_dir
        )
        image_widget.s_delete_image.connect(self.delete_image)
        self.image_list.addItem(image_item)
        self.image_list.setItemWidget(image_item, image_widget)

    def add_data(self, data: List[OCRData], cached=False):
        self.clear_data()

        size_hint = self.sizeHint()
        target_width = size_hint.width()-120

        for _data in data:
            self.add_image_widget(_data, target_width)

    def refresh_data(self, data: List[OCRData]):
        """
        This is called to refresh the list after an image has been manually deleted
        """
        self.clear_data()

        _sizeHint = self.sizeHint()
        _targetWidth = _sizeHint.width() - 80
        for _data in data:
            self.add_image_widget(_data, _targetWidth)

    def delete_image(self, guid: UUID):
        self.view_model.delete_image_by_guid(guid)

    def clear_data(self):
        self.image_list.clear()



class TextWidgetList(QListWidget):
    sign_on_selected_item = Signal(UUID)

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setObjectName("TextListWidget")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.v_scrollbar = QScrollBar(self)
        self.h_scrollbar = QScrollBar(self)
        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)

        self.v_scrollbar.setStyleSheet(
            """
            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }

            QScrollBar::handle:vertical {	
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #C80021;
            }

            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-line:vertical {
                height: 0px;
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }

            """
        )

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
        """
        )


class TextWidget(QWidget):
    s_update_label = Signal(OCRLine)
    """
    Custom widget holding the actual text data
    """

    def __init__(self, ocr_line: OCRLine, qfont: QFont, execution_dir: str):
        super().__init__()
        self.setObjectName("TextWidget")
        self.ocr_line = ocr_line
        self.qfont = qfont
        self.label = QLabel()
        self.label.setObjectName("TextLine")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setFont(qfont)
        self.label.setText(self.ocr_line.text)

        self.btn_edit_icon = os.path.join(execution_dir, "Assets", "Textures", "edit_icon.png")
        self.btn_edit = MenuButton("Edit Line", self.btn_edit_icon, 14, 14)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn_edit)
        self.setLayout(self.layout)

        # bind edit signal
        self.btn_edit.clicked.connect(self.edit_label)

    def edit_label(self):
        dialog = TextInputDialog("Editing Line", self.ocr_line.text, self.qfont, parent=self)

        if dialog.exec():
            new_text = dialog.new_text
            self.ocr_line.text = new_text
            self.label.setText(new_text)
            self.s_update_label.emit(self.ocr_line)


class TextView(QFrame):
    def __init__(self, platform: Platform, dataview: DataViewModel, execution_dir: str, font_path: str, font_size: int = 18, encoding: Encoding = Encoding.Unicode, translation_manager=None):
        super().__init__()
        self.setObjectName("TextView")
        self.setContentsMargins(0, 0, 0, 0)
        self.platform = platform
        self._dataview = dataview
        self.execution_dir = execution_dir
        self.translation_manager = translation_manager
        self.clip_board = QApplication.clipboard()

        # load persisted font size or default
        from PySide6.QtCore import QSettings
        settings = QSettings("BDRC", "TibetanOCRApp")
        self.font_size = settings.value("main/font_size", font_size, type=int)
        self.encoding = encoding
        self.default_font_path = font_path

        if self.platform == Platform.Windows:
            
            font_id = QFontDatabase.addApplicationFont(self.default_font_path)
            
            if font_id == -1:
                print("Failed to load font")
            else:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    tibetan_font_family = font_families[0]
                else:
                    tibetan_font_family = "Sans"  # Fallback font

                self.qfont = QFont(tibetan_font_family, self.font_size)
        else:
            self.qfont = QFont(self.default_font_path, self.font_size)

        self.page_guid = None
        self.ocr_lines = []
        self.current_font = ""
        self.text_widget_list = TextWidgetList()

        self.zoom_in_btn = TextToolsButton("+")
        self.zoom_out_btn = TextToolsButton("-")

        self.convert_wylie_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "convert_wylie_unicode.png")
        self.convert_wylie_btn = MenuButton(
            tr("convert between Wylie and Unicode"),
            self.convert_wylie_btn_icon,
            width=32,
            height=32,
            object_name="TextToolsButton"
        )

        self.copy_text_btn_icon = os.path.join(execution_dir, "Assets", "Textures", "copy.png")
        self.copy_text_btn = MenuButton(
            tr("copy all text lines"),
            self.copy_text_btn_icon,
            width=32,
            height=32,
            object_name="TextToolsButton"
        )

        self.spacer = QLabel()

        # bind signals
        self._dataview.s_data_selected.connect(self.handle_text_update)
        self._dataview.s_record_changed.connect(self.handle_text_update)
        self._dataview.s_ocr_line_update.connect(self.handle_line_update)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.convert_wylie_btn.clicked.connect(self.convert_wylie_unicode)
        self.copy_text_btn.clicked.connect(self.copy_text)

        # build layout
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.zoom_in_btn)
        self.button_layout.addWidget(self.zoom_out_btn)
        self.button_layout.addWidget(self.convert_wylie_btn)
        self.button_layout.addWidget(self.copy_text_btn)
        self.button_layout.addWidget(self.spacer)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.text_widget_list)
        self.setLayout(self.layout)

    def zoom_in(self):
        if len(self.ocr_lines) == 0:
            return

        self.text_widget_list.clear()
        new_size = self.qfont.pointSize() + 1
        self.qfont.setPointSize(new_size)
        # persist font size
        from PySide6.QtCore import QSettings
        QSettings("BDRC", "TibetanOCRApp").setValue("main/font_size", new_size)
        self.font_size = new_size

        if self.ocr_lines is not None:
            for ocr_line in self.ocr_lines:
                list_item = QListWidgetItem()

                text_widget = TextWidget(ocr_line, self.qfont, self.execution_dir)
                text_size = text_widget.sizeHint()
                text_widget.s_update_label.connect(self.handle_line_edit)

                if text_size.width() < 800:
                    list_item.setSizeHint(QSize(800, text_size.height()))
                else:
                    list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

                self.text_widget_list.addItem(list_item)
                self.text_widget_list.setItemWidget(list_item, text_widget)

            for idx in range(self.text_widget_list.count()):
                if idx % 2 == 0:
                    _qBrush = QBrush(QColor("#172832"))
                else:
                    _qBrush = QBrush(QColor("#1d1c1c"))
                self.text_widget_list.item(idx).setBackground(_qBrush)

    def zoom_out(self):
        if len(self.ocr_lines) == 0:
            return

        self.text_widget_list.clear()
        new_size = self.qfont.pointSize() - 1
        self.qfont.setPointSize(new_size)
        # persist font size
        from PySide6.QtCore import QSettings
        QSettings("BDRC", "TibetanOCRApp").setValue("main/font_size", new_size)
        self.font_size = new_size

        if self.ocr_lines is not None:
            for text_line in self.ocr_lines:
                list_item = QListWidgetItem()

                text_widget = TextWidget(text_line, self.qfont, self.execution_dir)
                text_size = text_widget.sizeHint()

                if text_size.width() < 800:
                    list_item.setSizeHint(QSize(800, text_size.height()))
                else:
                    list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

                self.text_widget_list.addItem(list_item)
                self.text_widget_list.setItemWidget(list_item, text_widget)

            for idx in range(self.text_widget_list.count()):
                if idx % 2 == 0:
                    _qBrush = QBrush(QColor("#172832"))
                else:
                    _qBrush = QBrush(QColor("#1d1c1c"))
                self.text_widget_list.item(idx).setBackground(_qBrush)

    def handle_text_update(self, ocr_data: OCRData):
        self.update_text(ocr_data.guid, ocr_data.ocr_lines)

    def update_text(self, page_guid: UUID, ocr_lines: List[OCRLine]):
        self.page_guid = page_guid
        self.ocr_lines = ocr_lines
        self.text_widget_list.clear()

        if ocr_lines is not None:
            for ocr_line in ocr_lines:
                list_item = QListWidgetItem()
                text_widget = TextWidget(ocr_line, self.qfont, self.execution_dir)

                text_size = text_widget.sizeHint()

                if text_size.width() < 800:
                    list_item.setSizeHint(QSize(800, text_size.height()))
                else:
                    list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

                self.text_widget_list.addItem(list_item)
                self.text_widget_list.setItemWidget(list_item, text_widget)

            for idx in range(self.text_widget_list.count()):
                if idx % 2 == 0:
                    _qBrush = QBrush(QColor("#172832"))
                else:
                    _qBrush = QBrush(QColor("#1d1c1c"))
                self.text_widget_list.item(idx).setBackground(_qBrush)

    def update_font(self, font_path: str):
        self.current_font = font_path

    def update_font_size(self, font_size: int):
        self.font_size = font_size

    def handle_line_edit(self, ocr_line: OCRLine):
        ocr_line_update = OCRLineUpdate(
            self.page_guid,
            ocr_line
        )
        self._dataview.update_ocr_line(ocr_line_update)

    def handle_line_update(self, ocr_data: OCRData):
        self.update_text(ocr_data.guid, ocr_data.ocr_lines)

    def convert_wylie_unicode(self):
        if self.page_guid is not None:
            self._dataview.convert_wylie_unicode(self.page_guid)

    def copy_text(self):
        self.clip_board.clear(mode=self.clip_board.Mode.Clipboard)

        clipboard_text = ""
        for ocr_line in self.ocr_lines:
            clipboard_text += f"{ocr_line.text}\n"

        self.clip_board.setText(clipboard_text, mode=self.clip_board.Mode.Clipboard)
        # position popover under copy button
        btn = self.copy_text_btn
        pos = btn.mapToGlobal(btn.rect().bottomLeft())
        QToolTip.showText(pos, "Copied to clipboard", btn)
