import uuid
import cv2
from PIL import Image
from typing import List
from PIL.ImageQt import ImageQt, QImage
from PySide6.QtGui import QPixmap, QTransform
from BDRC.Data import Line
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsItem

from BDRC.line_detection import rotate_from_angle


class ImagePreview(QGraphicsPixmapItem):
    def __init__(self, image_path: str, lines: List[Line], angle: float):
        super().__init__()
        self.image_path = image_path
        self.image = Image.open(self.image_path)
        self.lines = lines
        self.angle = angle
        self.guid = uuid.uuid1() # check if that is really ok, or the original data guid should be passed
        self.is_in_preview = False

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        self.show_image()

    def show_image(self):
        q_image = QImage(self.image_path)
        pixMap = QPixmap.fromImage(q_image)

        self.setPixmap(pixMap)
        self.is_in_preview = False
        self.update()

    def show_preview(self):
        line_preview = cv2.imread(self.image_path)
        line_preview = cv2.cvtColor(line_preview, cv2.COLOR_BGR2RGB)
        line_preview = rotate_from_angle(line_preview, self.angle)

        color = (255, 100, 0)

        if self.lines is not None and len(self.lines) > 0:
            for idx, line in enumerate(self.lines):
                cv2.drawContours(
                    line_preview, [line.contour], contourIdx=-1, color=color, thickness=4
                )
            preview = Image.fromarray(line_preview)
            q_image = ImageQt(preview)
            pixmap = QPixmap.fromImage(q_image) # https://pillow.readthedocs.io/en/stable/reference/ImageQt.html
            self.setPixmap(pixmap)
            self.is_in_preview = True
            self.update()