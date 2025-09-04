"""
Export functionality for OCR results in various formats.

This module provides classes for exporting OCR results to different formats
including plain text, PageXML, and JSON. The base Exporter class defines
the interface, with concrete implementations for each format.
"""

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
from BDRC.line_detection import optimize_countour
from BDRC.Utils import (
    get_utc_time,
    rotate_contour,
    get_text_bbox,
)



class Exporter:
    """
    Abstract base class for OCR result exporters.
    
    Defines the interface and common functionality for exporting OCR results
    to various formats. Subclasses implement specific export formats.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the exporter with output directory.
        
        Args:
            output_dir: Directory path where exported files will be saved
        """
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
        """
        Convert BBox object to tuple format.
        
        Args:
            bbox: BBox object with x, y, w, h attributes
            
        Returns:
            Tuple of (x, y, width, height)
        """
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return x, y, w, h

    @staticmethod
    def get_text_points(contour):
        """
        Convert contour points to string format for XML export.
        
        Args:
            contour: Contour points array
            
        Returns:
            Space-separated string of x,y coordinates
        """
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    @staticmethod
    def get_bbox_points(bbox: BBox):
        """
        Convert BBox to coordinate points string for XML export.
        
        Args:
            bbox: BBox object defining rectangular region
            
        Returns:
            String of corner coordinates in XML format
        """
        points = f"{bbox.x},{bbox.y} {bbox.x + bbox.w},{bbox.y} {bbox.x + bbox.w},{bbox.y + bbox.h} {bbox.x},{bbox.y + bbox.h}"
        return points


class PageXMLExporter(Exporter):
    """
    Exporter for PageXML format compatible with Transkribus and other OCR tools.
    
    PageXML is a standardized format for representing document layout and
    OCR results with detailed coordinate information.
    """
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize PageXML exporter.
        
        Args:
            output_dir: Directory path for exported XML files
        """
        super().__init__(output_dir)
        logging.info("Init XML Exporter")

    def get_text_line_block(self, coordinate, index: int, unicode_text: str):
        """
        Create XML element for a single text line.
        
        Args:
            coordinate: Line coordinate information
            index: Line index for ordering
            unicode_text: Recognized text content
            
        Returns:
            XML element representing the text line
        """
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
        """
        Build complete PageXML document structure.
        
        Args:
            image: Source image array for dimensions
            image_name: Name of the image file
            text_bbox: Bounding box coordinates for text region
            lines: List of line coordinate strings
            text_lines: List of OCR text results
            
        Returns:
            Formatted XML document string
        """
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
    """
    Simple text file exporter for OCR results.
    
    Exports recognized text as plain text files with one line per text line.
    """
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize text exporter.
        
        Args:
            output_dir: Directory path for exported text files
        """
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
        """
        Export OCR results to plain text file.
        
        Args:
            image: Source image (not used for text export)
            image_name: Base name for output file
            lines: Line detection results (not used for text export)
            text_lines: OCR text results to export
            optimize: Optimization flag (not used for text export)
            bbox: Bounding box flag (not used for text export)
            angle: Rotation angle (not used for text export)
        """

        out_file = f"{self.output_dir}/{image_name}.txt"

        with open(out_file, "w", encoding="UTF-8") as f:
            for _line in text_lines:
                f.write(f"{_line.text}\n")

    def export_text(self, image_name: str, lines: List[OCRLine]):
        """
        Export text lines to a plain text file.
        
        Args:
            image_name: Base name for output file
            lines: List of OCR text lines to export
        """
        out_file = f"{self.output_dir}/{image_name}.txt"

        with open(out_file, "w", encoding="UTF-8") as f:
            for _line in lines:
                f.write(f"{_line.text}\n")


class JsonExporter(Exporter):
    """
    JSON/JSONL exporter for OCR results with coordinate information.
    
    Exports OCR results as JSON Lines format including both text content
    and coordinate information for lines and text regions.
    """
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize JSON exporter.
        
        Args:
            output_dir: Directory path for exported JSON files
        """
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
        """
        Export OCR results to JSON Lines format.
        
        Args:
            image: Source image (used for coordinate transformation)
            image_name: Base name for output file
            lines: Line detection results with coordinates
            text_lines: OCR text results
            optimize: Whether to optimize contours
            bbox: Whether to use bounding boxes instead of contours
            angle: Rotation angle for coordinate transformation
        """

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
