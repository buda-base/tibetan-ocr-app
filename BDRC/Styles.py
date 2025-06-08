import resources


DARK = """
    /* Global button styles */

    QPushButton:hover {
        border-color: #ffad00;
    }

    QFrame#ImageGallery {
        background-color: #1d1c1c;
    }

    QListWidget#ImageGalleryList {
        background-color: #100f0f;
        border: 4px solid #100f0f;

        QListWidget {
            color: #ffffff;
            background-color: #100f0f;
            border: 4px solid #100f0f;
        }

        QListWidget::item:selected {
            background: #2d2d46;
        }
    }

    QWidget#MainWindow {
        background-color: #1d1c1c;
        color: #000000;
    }

    QFrame#MainCanvas {
        color: #ffffff;
        background-color: #100F0F;
        border: 1px solid #ffad00;
        border-radius: 2px;
    }

    QFrame#HeaderTools {
        background-color: #100F0F;
        margin-left: 0px;
        alignment: left;
        border: 2px solid #100F0F;
        border-radius: 6px;
    }

    QFrame#PageSwitcher {
        color: #ffffff;
        border: 2px solid #100F0F;
        border-radius: 8px;
    }

    QFrame#TextView {
        color: #ffffff;
        background-color: #100F0F;
        border: 1px solid #ffad00;
        border-radius: 2px;
    }

    QWidget#ToolBox {
        color: #ffffff;
        background-color: #100F0F;
        alignment-left;
    }

    QComboBox#ModelSelection {
        background: #434343;
        color: #ffffff;
        min-width: 220px;
        border: 2px solid #ced4da;
        border-radius: 4px;
        padding: 4px 4px 4px 4px;
    }

    QComboBox#ModelSelection:hover {
        border: 2px solid #ffad00;
    }

    QPushButton#MenuButton {
        background-color: #172832;
        border: 1px solid #1d1d1d;
        border-radius: 4px;
    }

    QPushButton#MenuButton:hover {
        border: 1px solid #ffad00;
    }

    QPushButton#CanvasToolButton {
        background-color: #3f3f3f;
        border: 2px solid #1d1d1d;
        border-radius: 4px;
        padding: 4px 4px 4px;
    }

    QPushButton#CanvasToolButton:hover {
        border: 2px solid #ffad00;
    }

    QPushButton#TextToolsButton {
        color: #ffffff;
        background-color: #3f3f3f;
        border: 2px solid #1d1d1d;
        border-radius: 4px;
    }

    QPushButton#TextToolsButton:hover {
        color: #ffad00;
        border: 2px solid #ffad00;
    }

    QDialog#ExportDialog {
        color: #ffffff;
        background-color: #1d1c1c;
    }

    QDialog#TextInputDialog {
        background-color: #1d1c1c;
    }

    QDialog#BatchOCRDialog {
        color: #ffffff;
        background-color: #1d1c1c;

        QLineEdit {
                color: #ffffff;
                background-color: #474747;
                border: 2px solid #343942;
                border-radius: 8px;
                padding: 6px;
                text-align: left;
            }
    }

    QDialog#SettingsDialog {
        background-color: #172832;

        QPushButton {
                color: #A40021;
                background-color: #fce08d;
                border-radius: 4px;
                height: 18;
            }

            QPushButton::hover {
                color: #ffad00;
                border: 1px solid #ffad00;
            }
    }

    QTabWidget::pane {
        background-color: #242424;
        border-width: 0px;  
        border-radius: 6px;
    }

    QTabWidget::tab-bar {
        left: 5px;
    }

    QTabBar::tab {
        color: #ffffff;
        background: #A40021;
        padding: 4px;
    }

    QTabWidget::tab-bar {
        left: 5px;
    }

    QTabBar::tab:selected {
        background: #730017;
        margin-bottom: -1px;
    }  

    QPushButton#DialogButton {
        color: #ffffff;
        background-color: #A40021;
        border-radius: 4px;
        height: 24;
    }

    QPushButton#DialogButton:hover {
        color: #ffad00;
        border: 1px solid #ffad00;
    }

    QPushButton#SmallDialogButton {
        color: #ffffff;
        background-color: #A40021;
        border-radius: 4px;
        height: 20;
        width: 80px;
    }

    QPushButton#SmallDialogButton:hover {
        color: #ffad00;
        border: 1px solid #ffad00;
    }

    QRadioButton#OptionsRadio {
        color: #ffffff;
        border: none;
    }

    QRadioButton#OptionsRadio:indicator:checked {
        image:url(:/Assets/Textures/qradio_indicator_checked.png);
        width: 12px;
        height: 12px;
    }

    QRadioButton#OptionsRadio:indicator:unchecked {
        image:url(:/Assets/Textures/qradio_indicator.png);
        width: 12px;
        height: 12px;
    }

    QLabel#DefaultLabel {
        color: #ffffff;
    }

    QLabel#OptionsLabel {
        color: #ffffff;
        width: 100px;
    }

    QLabel#OptionsExplanation {
        color: #CCCCCC;
        font-style: italic;
        width: 100%;
    }

    QLabel#HeaderLabel {
        color: #ffffff;
        background-color: #172832;
        font-weight: bold;
        padding: 4px 4px 4px 4px;
        width: 80px;
    }

    QLabel#PageNumberLabel {
        color: #ffffff;
        font-weight: bold;
        background: #434343;
        border: 1px solid white;
        border-radius: 4px;
    }

    QLineEdit#DialogLineEdit {
        color: #ffffff;
        background-color: #474747;
        border: 2px solid #343942;
        border-radius: 2px;
        text-align: left;
    }

    QProgressBar#DialogProgressBar {
        background-color: #474747;
        color: #A40021;
        border: 2px solid #343942;
        border-radius: 8px;
        padding: 4px 4px 4px 4px;
    }

    QProgressBar#DialogProgressBar::chunk {
        background-color: #fcc104;
        border-radius: 5px;
        margin: 3px 3px 3px 3px;
    }

    QWidget#QModelList {
        color: #ffffff;
        width: 80%;
    }

    QListWidget#ModelList {
        border 4px solid yellow;
        background-color: #464646;
    }

    QListWidget#TextListWidget {
        color: #ffffff;
        background-color: #172832;
    }

    QWidget#TextWidget {
        color: #ffffff;
    }

    QLabel#TextLine {
        color: #ffffff;
    }

    QGraphicsView#PTGraphicsView {
        background-color: #172832;
    }
    
    QScrollBar:vertical {
        border: none;
        background: #2d2d46;
        width: 25px;
        margin: 10px 5px 15px 10px;
        border-radius: 0px;
    }

    QSplitter {
        background-color: #100F0F;
    }

    QSplitter::handle {
        padding-top: 4px;
        padding-bottom: 4px;
        image: url(:/Assets/Textures/splitter_handle.png);
    }

    QSplitter::handle:pressed {
        image: url(:/Assets/Textures/splitter_handle_press.png);
    }

    QTableWidget#ModelTable {
        color: #ffffff;
        background-color: #172832;
    }

    QMessageBox#NotificationWindow {
        color: #ffffff;
        background-color: #1d1c1c;
                    
        QPushButton {
            width: 200px;
            padding: 5px;
            background-color: #A40021;
            border-radius: 4px;
        }

        QPushButton:hover {
            border: 1px solid #ffad00;
            color: #ffad00;
        }
    }

    QAbstractScrollArea::corner {
        background: none;
        border: none;
    }

"""
