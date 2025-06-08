import abc
import json
import os
from io import TextIOBase
from typing import TextIO
import pyewts
import logging
import numpy.typing as npt
from xml.dom import minidom
import xml.etree.ElementTree as etree
from BDRC.Data import BBox, OCRData, ExportSettings, ExportFormat, ExportFileMode, TransformedContours
from BDRC.Utils import (
    get_utc_time,
    rotate_contour,
    optimize_countour, get_text_bbox,
)



class Exporter(abc.ABC):
    def __init__(self, settings: ExportSettings):
        self.settings = settings
        self.converter = pyewts.pyewts()
        logging.info("Init Exporter")

    @staticmethod
    def create_exporter(settings: ExportSettings) -> 'Exporter':
        format = settings.format
        exporter: Exporter

        if format == ExportFormat.Text:
            exporter = TextExporter(settings)
        elif format == ExportFormat.PageXML:
            exporter = PageXMLExporter(settings)
        elif format == ExportFormat.JSONLines:
            exporter = JsonLinesExporter(settings)
        else:
            raise ValueError(f"Invalid export format: '{format}'")

        return exporter

    def export(
        self,
        data: list[OCRData]
    ):
        """ Exports the OCR data. Whether the output will be separate files per page,
         or one big file, depends on `ExportSettings.file_mode`.
        """
        file_mode = self.settings.file_mode

        if file_mode == ExportFileMode.FilePerPage:
            for ocr_data in data:
                self._export_page(None, ocr_data)

        elif file_mode == ExportFileMode.OneCombinedFile:
            self._export_one_big_file(data)

        else:
            raise ValueError(f"Invalid export file mode: '{file_mode}'")

    @abc.abstractmethod
    def _export_one_big_file(
        self,
        data: list[OCRData],
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):
        """ Exports text ocr_lines and line information of all pages into one big file """
        raise NotImplementedError

    @abc.abstractmethod
    def _export_page(
        self,
        image: npt.NDArray | None,
        ocr_data: OCRData
    ):
        """ Exports text ocr_lines and line information """
        raise NotImplementedError

    def _transform_contours(
            self,
            image: npt.NDArray | None,
            ocr_data: OCRData,
            optimize: bool = True,
            bbox: bool = False,
            angle: float = 0.0
    ) -> TransformedContours:
        """ Transforms and optionally optimized the OCR data's contours. Mutates the `OCRLines` directly,
        and returns the plain lines and plain BBox.
        """

        if image is not None and angle != 0:
            x_center = image.shape[1] // 2
            y_center = image.shape[0] // 2

            for line in ocr_data.lines:
                line.contour = rotate_contour(
                    line.contour, (x_center, y_center), angle
                )

        if optimize:
            for line in ocr_data.lines:
                line.contour = optimize_countour(line.contour)

        if bbox:
            plain_lines = [self.get_bbox(x.bbox) for x in ocr_data.lines]
        else:
            plain_lines = [self.get_text_points(x.contour) for x in ocr_data.lines]

        text_bbox = get_text_bbox(ocr_data.lines)
        plain_box = self.get_bbox_points(text_bbox)

        return TransformedContours(plain_box, plain_lines)

    @staticmethod
    def get_bbox(bbox: BBox) -> tuple[int, int, int, int]:
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return x, y, w, h

    @staticmethod
    def get_text_points(contour: npt.NDArray) -> str:
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    @staticmethod
    def get_bbox_points(bbox: BBox):
        points = f"{bbox.x},{bbox.y} {bbox.x + bbox.w},{bbox.y} {bbox.x + bbox.w},{bbox.y + bbox.h} {bbox.x},{bbox.y + bbox.h}"
        return points

    def _prepare_surrounding_texts(self, image_name: str) -> tuple[str, str]:
        def resolve_vars(text: str):
            # Turn \n into a real newline
            text = text.replace("\\n", "\n")
            text = text.replace("{image}", image_name)
            return text

        return (resolve_vars(self.settings.before_page_text), resolve_vars(self.settings.after_page_text))

    def _surround_text(self, before_text: str, text: str, after_text: str) -> str:
        """ The simplest way to handle before/after page texts is to simply concatenate them to the
        OCR'd text. If a before/after page text is defined, a newline separates it from the OCR'd text.
        """
        before_text = f"{before_text}\n" if before_text else ""
        after_text = f"\n{after_text}" if after_text else ""
        return f"{before_text}{text}{after_text}"


class PageXMLExporter(Exporter):
    def __init__(self, settings: ExportSettings) -> None:
        super().__init__(settings)
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
        pages: list[tuple[npt.NDArray | None, OCRData]],
        optimize: bool,
        bbox: bool,
        angle: float
    ) -> str:
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

        for (image, ocr_data) in pages:
            transformed_contours = self._transform_contours(image, ocr_data, optimize, bbox, angle)
            self._build_xml_page(root, image, ocr_data, transformed_contours)

        parsed_xml = minidom.parseString(etree.tostring(root))
        parsed_xml = parsed_xml.toprettyxml()

        return parsed_xml

    def _build_xml_page(
        self,
        root: etree.Element,
        image: npt.NDArray | None,
        ocr_data: OCRData,
        transformed_contours: TransformedContours
    ) -> None:
        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = ocr_data.image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}" if image else ""
        page.attrib["imageHeight"] = f"{image.shape[0]}" if image else ""

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
        text_region_coords.attrib["points"] = transformed_contours.plain_box

        print(f"Exporting XML Lines: {len(ocr_data.ocr_lines)}")
        print(f"Exporting Line Info: {len(transformed_contours.plain_lines)}")

        for l_idx, line in enumerate(transformed_contours.plain_lines):
            unicode_text = ocr_data.ocr_lines[l_idx].text if ocr_data.ocr_lines else ""

            text_region.append(self.get_text_line_block(coordinate=line, index=l_idx, unicode_text=unicode_text))

    def _export_one_big_file(
            self,
            data: list[OCRData],
            optimize: bool = True,
            bbox: bool = False,
            angle: float = 0.0
    ):
        out_file = self.settings.output_file
        xml_doc = self.build_xml_document([(None, ocr_data) for ocr_data in data], optimize, bbox, angle)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(xml_doc)

    def _export_page(
        self,
        image: npt.NDArray | None,
        ocr_data: OCRData,
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):
        out_file = f"{self.settings.output_dir}/{ocr_data.image_name}.xml"
        xml_doc = self.build_xml_document([(image, ocr_data)], optimize, bbox, angle)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(xml_doc)


class TextExporter(Exporter):
    def __init__(self, settings: ExportSettings) -> None:
        super().__init__(settings)
        logging.info("Init Text Exporter")

    def _export_one_big_file(
        self,
        data: list[OCRData],
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):
        out_file = self.settings.output_file

        with open(out_file, "w", encoding="UTF-8") as f:
            for ocr_data in data:
                before_text, after_text = self._prepare_surrounding_texts(ocr_data.image_name)
                self._write_page_to_file(f, ocr_data, before_text, after_text)

    def _export_page(
            self,
            image: npt.NDArray | None,
            ocr_data: OCRData,
            optimize: bool = True,
            bbox: bool = False,
            angle: float = 0.0
    ):
        out_file = os.path.join(self.settings.output_dir, f"{ocr_data.image_name}.txt")
        before_text, after_text = self._prepare_surrounding_texts(ocr_data.image_name)

        with open(out_file, "w", encoding="UTF-8") as f:
            self._write_page_to_file(f, ocr_data, before_text, after_text)

    def _write_page_to_file(self, f: TextIO, ocr_data: OCRData, before_text: str, after_text: str):
        if before_text:
            f.write(before_text)
            f.write("\n")

        for ocr_line in ocr_data.ocr_lines:
            f.write(ocr_line.text)
            f.write("\n")

        if after_text:
            f.write("\n")
            f.write(after_text)


class JsonLinesExporter(Exporter):
    def __init__(self, settings: ExportSettings) -> None:
        super().__init__(settings)
        logging.info("Init JSONLines Exporter")

    def _export_one_big_file(
        self,
        data: list[OCRData],
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):
        out_file = self.settings.output_file

        with open(out_file, "w", encoding="UTF-8") as f:
            if data:
                for ocr_data in data:
                    json_record = self._to_json_record(None, ocr_data, optimize, bbox, angle)
                    self._write_to_file(f, json_record)

                    if ocr_data is not data[-1]:
                        f.write("\n")


    def _export_page(
        self,
        image: npt.NDArray | None,
        ocr_data: OCRData,
        optimize: bool = True,
        bbox: bool = False,
        angle: float = 0.0
    ):
        out_file = f"{self.settings.output_dir}/{ocr_data.image_name}.jsonl"

        json_record = self._to_json_record(image, ocr_data, optimize, bbox, angle)

        with open(out_file, "w", encoding="UTF-8") as f:
            self._write_to_file(f, json_record)

    def _to_json_record(self, image: npt.NDArray | None, ocr_data: OCRData, optimize: bool, bbox: bool, angle: float) -> dict:
        transformed_contours = self._transform_contours(image, ocr_data, optimize, bbox, angle)

        _text_lines = [l.text for l in ocr_data.ocr_lines]

        json_record = {
            "image": ocr_data.image_name,
            "textbox": transformed_contours.plain_box,
            "lines": transformed_contours.plain_lines,
            "text": _text_lines,
        }

        return json_record

    def _write_to_file(self, f: TextIOBase, json_record: dict):
        # Don't specify 'indent'. We don't want to pretty-print so that we're JSONLines-compatible.
        json_line = json.dumps(json_record, ensure_ascii=False)
        f.write(json_line)
