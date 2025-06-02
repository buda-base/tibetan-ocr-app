import abc
import json
from typing import List
import pyewts
import logging
import numpy as np
import numpy.typing as npt
from xml.dom import minidom
import xml.etree.ElementTree as etree
from BDRC.Data import BBox, Line, OCRLine
from BDRC.Utils import (
    get_utc_time,
    rotate_contour,
    optimize_countour, get_text_bbox,
)



class Exporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.converter = pyewts.pyewts()
        logging.info("Init Exporter")


    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "export_lines")
            and callable(subclass.export_lines)
            or NotImplemented
        )

    @abc.abstractmethod
    def export_text(self, image_name: str, text_lines: List[str]):
        """ Exports only the text lines """
        raise NotImplementedError

    @abc.abstractmethod
    def export_lines(
        self,
        image: npt.NDArray | None,
        image_name: str,
        lines: List[Line],
        text_lines: List[str],
    ):
        """ Exports text lines and line informations """
        raise NotImplementedError

    @staticmethod
    def get_bbox(bbox: BBox) -> tuple[int, int, int, int]:
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return x, y, w, h

    @staticmethod
    def get_text_points(contour):
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    @staticmethod
    def get_bbox_points(bbox: BBox):
        points = f"{bbox.x},{bbox.y} {bbox.x + bbox.w},{bbox.y} {bbox.x + bbox.w},{bbox.y + bbox.h} {bbox.x},{bbox.y + bbox.h}"
        return points


class PageXMLExporter(Exporter):
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)
        logging.info("Init XML Exporter")

    def get_text_line_block(self, coordinate, index: int, unicode_text: str):
        text_line = etree.Element(
            "Textline", id="", custom=f"readingOrder {{index:{index};}}"
        )
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords

        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line

    def build_xml_document(
        self,
        image: npt.NDArray,
        image_name: str,
        text_bbox: str,
        lines: List[str],
        text_lines: List[OCRLine] | None,
    ):
        root = etree.Element("PcGts")
        root.attrib["xmlns"] = (
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        )
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib["xsi:schemaLocation"] = (
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
        )

        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = get_utc_time()

        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = text_bbox

        print(f"Exporting XML Lines: {len(text_lines)}")
        print(f"Exporting Line Info: {len(lines)}")

        for l_idx, line in enumerate(lines):
            if text_lines is not None and len(text_lines) > 0:
                text_region.append(
                    self.get_text_line_block(
                        coordinate=line, index=l_idx, unicode_text=text_lines[l_idx].text
                    )
                )
            else:
                text_region.append(
                    self.get_text_line_block(
                        coordinate=line, index=l_idx, unicode_text=""
                    )
                )

        parsed_xml = minidom.parseString(etree.tostring(root))
        parsed_xml = parsed_xml.toprettyxml()

        return parsed_xml

    def export_lines(
        self,
        image: npt.NDArray | None,
        image_name: str,
        lines: List[Line],
        text_lines: List[OCRLine],
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):

        if angle != abs(0):
            x_center = image.shape[1] // 2
            y_center = image.shape[0] // 2

            for line in lines:
                line.contour = rotate_contour(
                    line.contour,
                    (x_center, y_center),
                    angle
                )

        if optimize:
            for line in lines:
                line.contour = optimize_countour(line.contour)

        if bbox:
            plain_lines = [self.get_bbox(x.bbox) for x in lines]
        else:
            plain_lines = [self.get_text_points(x.contour) for x in lines]

        text_bbox = get_text_bbox(lines)
        plain_box = self.get_bbox_points(text_bbox)

        xml_doc = self.build_xml_document(
            image,
            image_name,
            text_bbox=plain_box,
            lines=plain_lines,
            text_lines=text_lines,
        )

        out_file = f"{self.output_dir}/{image_name}.xml"

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(xml_doc)

class TextExporter(Exporter):
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)
        logging.info("Init Text Exporter")

    def export_lines(
            self,
            image: npt.NDArray | None,
            image_name: str,
            lines: List[Line],
            text_lines: list[OCRLine],
            optimize: bool = True,
            bbox: bool = False,
            angle: float = 0.0):

        out_file = f"{self.output_dir}/{image_name}.txt"

        with open(out_file, "w", encoding="UTF-8") as f:
            for _line in text_lines:
                f.write(f"{_line.text}\n")

    def export_text(self, image_name: str, lines: List[OCRLine]):
        out_file = f"{self.output_dir}/{image_name}.txt"

        with open(out_file, "w", encoding="UTF-8") as f:
            for _line in lines:
                f.write(f"{_line.text}\n")


class JsonExporter(Exporter):
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)
        logging.info("Init JSON Exporter")

    def export_lines(
        self,
        image: npt.NDArray | None,
        image_name: str,
        lines: List[Line],
        text_lines: list[OCRLine],
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):

        if angle != abs(0):
            x_center = image.shape[1] // 2
            y_center = image.shape[0] // 2

            for line in lines:
                line.contour = rotate_contour(
                    line.contour, (x_center, y_center), angle
                )

        if optimize:
            for line in lines:
                line.contour = optimize_countour(line.contour)

        if bbox:
            plain_lines = [self.get_bbox(x.bbox) for x in lines]
        else:
            plain_lines = [self.get_text_points(x.contour) for x in lines]

        text_bbox = get_text_bbox(lines)
        plain_box = self.get_bbox_points(text_bbox)
        _text_lines = [x.text for x in text_lines]
        json_record = {
            "image": image_name,
            "textbox": plain_box,
            "lines": plain_lines,
            "text": _text_lines,
        }

        out_file = f"{self.output_dir}/{image_name}.jsonl"

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(json_record, f, ensure_ascii=False, indent=1)
