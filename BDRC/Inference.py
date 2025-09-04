import cv2
import pyewts
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from typing import List, Union


from scipy.special import softmax
from Config import COLOR_DICT, CHARSETENCODER
from BDRC.Data import (
    OCRLine,
    OpStatus,
    TPSMode,
    Encoding,
    OCRModelConfig,
    LineDetectionConfig,
    LayoutDetectionConfig, Platform, CharsetEncoder
)

from pyctcdecode import build_ctcdecoder
from BDRC.line_detection import (
    build_line_data,
    extract_line_images,
    optimize_countour,
    sort_lines_by_threshold2,
    build_raw_line_data,
    filter_line_contours
)

from BDRC.image_dewarping import (
    apply_global_tps,
    check_for_tps
)

from BDRC.Utils import (
    preprocess_image,
    binarize,
    normalize,
    stitch_predictions,
    tile_image,
    sigmoid,
    pad_to_height,
    pad_to_width,
    get_execution_providers
)


class CTCDecoder:
    def __init__(self, charset: str | List[str], add_blank: bool):

        if isinstance(charset, str):
            self.charset = [x for x in charset]

        elif isinstance(charset, List):
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        self.add_blank = add_blank

        if self.add_blank:
            self.ctc_vocab.insert(0, " ")
        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def encode(self, label: str):
        return [self.charset.index(x) + 1 for x in label]

    def decode(self, inputs: List[int]) -> str:
        return "".join(self.charset[x - 1] for x in inputs)

    def ctc_decode(self, logits):
        return self.ctc_decoder.decode(logits).replace(" ", "")


class Detection:
    def __init__(self, platform: Platform, config: LineDetectionConfig | LayoutDetectionConfig):
        self.platform = platform
        self.config = config
        self._config_file = config
        self._onnx_model_file = config.model_file
        self._patch_size = config.patch_size
        self._execution_providers = get_execution_providers()
        self._inference = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )

    def _preprocess_image(self, image: npt.NDArray, patch_size: int = 512):
        padded_img, pad_x, pad_y = preprocess_image(image, patch_size)
        tiles, y_steps = tile_image(padded_img, patch_size)
        tiles = [binarize(x) for x in tiles]
        tiles = [normalize(x) for x in tiles]
        tiles = np.array(tiles)

        return padded_img, tiles, y_steps, pad_x, pad_y

    def _crop_prediction(
            self, image: npt.NDArray, prediction: npt.NDArray, x_pad: int, y_pad: int
    ) -> npt.NDArray:
        x_lim = prediction.shape[1] - x_pad
        y_lim = prediction.shape[0] - y_pad

        prediction = prediction[:y_lim, :x_lim]
        prediction = cv2.resize(prediction, dsize=(image.shape[1], image.shape[0]))

        return prediction

    def _predict(self, image_batch: npt.NDArray):
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()

        return prediction

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        pass


class LineDetection(Detection):
    def __init__(self, platform: Platform, config: LineDetectionConfig) -> None:
        super().__init__(platform, config)

    def predict(self, image: npt.NDArray, class_threshold: float = 0.9) -> npt.NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size)
        prediction = self._predict(tiles)
        prediction = np.squeeze(prediction, axis=1)
        prediction = sigmoid(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class LayoutDetection(Detection):
    def __init__(self, platform: Platform, config: LayoutDetectionConfig, debug: bool = False) -> None:
        super().__init__(platform, config)
        self._classes = config.classes
        self._debug = debug

    def _get_contours(self, prediction: npt.NDArray, optimize: bool = True, size_tresh: int = 200) -> List:
        prediction = np.where(prediction > 200, 255, 0)
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            if optimize:
                contours = [optimize_countour(x) for x in contours]
                contours = [x for x in contours if cv2.contourArea(x) > size_tresh]
            return contours
        else:
            return []

    def create_preview_image(self,
                             image: npt.NDArray,
                             prediction: npt.NDArray,
                             alpha: float = 0.4,
                             ) -> npt.NDArray | None:

        if image is None:
            return None

        image_predictions = self._get_contours(prediction[:, :, 1])
        line_predictions = self._get_contours(prediction[:, :, 2])
        caption_predictions = self._get_contours(prediction[:, :, 3])
        margin_predictions = self._get_contours(prediction[:, :, 4])

        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(line_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size)
        prediction = self._predict(tiles)
        prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
        prediction = softmax(prediction, axis=-1)
        prediction = np.where(prediction > class_threshold, 1.0, 0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class OCRInference:
    def __init__(self, platform: Platform, ocr_config: OCRModelConfig):
        self.platform = platform
        self.config = ocr_config
        self._onnx_model_file = ocr_config.model_file
        self._input_width = ocr_config.input_width
        self._input_height = ocr_config.input_height
        self._input_layer = ocr_config.input_layer
        self._output_layer = ocr_config.output_layer
        self._characters = ocr_config.charset
        self._squeeze_channel_dim = ocr_config.squeeze_channel
        self._swap_hw = ocr_config.swap_hw
        self._execution_providers = get_execution_providers()
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )
        self._add_blank = ocr_config.add_blank
        self.decoder = CTCDecoder(self._characters, self._add_blank)

    def _pad_ocr_line(
            self,
            img: npt.NDArray,
            padding: str = "black",
    ) -> npt.NDArray:

        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio < height_ratio:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        elif width_ratio > height_ratio:
            out_img = pad_to_height(img, self._input_width, self._input_height, padding)
        else:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        return cv2.resize(
            out_img,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _prepare_ocr_line(self, image: npt.NDArray) -> npt.NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = binarize(line_image)

        if len(line_image.shape) == 3:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)

        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        line_image = line_image.astype(np.float32)

        return line_image

    def _pre_pad(self, image: npt.NDArray):
        """
        Adds a small white patch of size HxH to the left and right of the line
        """
        h, _, c = image.shape
        patch = np.ones(shape=(h, h, c), dtype=np.uint8)
        patch *= 255
        out_img = np.hstack(tup=[patch, image, patch])
        return out_img

    def _predict(self, image_batch: npt.NDArray) -> npt.NDArray:
        image_batch = image_batch.astype(np.float32)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )

        logits = ocr_results[0].numpy()
        logits = np.squeeze(logits)

        return logits

    def _decode(self, logits: npt.NDArray) -> str:
        if logits.shape[0] == len(self.decoder.ctc_vocab):
            logits = np.transpose(
                logits, axes=[1, 0]
            )  # adjust logits to have shape time, vocab

        text = self.decoder.ctc_decode(logits)

        return text

    def run(self, line_image: npt.NDArray, pre_pad: bool = True) -> str:

        if pre_pad:
            line_image = self._pre_pad(line_image)
        line_image = self._prepare_ocr_line(line_image)

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

        logits = self._predict(line_image)
        text = self._decode(logits)

        return text


class OCRPipeline:
    """
    Note: The handling of line model vs. layout model is kind of provisional here and totally depends on the way you want to run this.
    You could also pass both configs to the pipeline, run both models and merge the (partially) overlapping output before extracting the line images to compensate for the strengths/weaknesses
    of either model. So that is basically up to you.
    """

    def __init__(
            self,
            platform: Platform,
            ocr_config: OCRModelConfig,
            line_config: LineDetectionConfig | LayoutDetectionConfig
    ):
        self.ready = False
        self.platform = platform
        self.ocr_model_config = ocr_config
        self.line_config = line_config
        self.encoder = ocr_config.encoder
        self.ocr_inference = OCRInference(self.platform, self.ocr_model_config)
        self.converter = pyewts.pyewts()

        if isinstance(self.line_config, LineDetectionConfig):
            self.line_inference = LineDetection(self.platform, self.line_config)
            self.ready = True
        elif isinstance(self.line_config, LayoutDetectionConfig):
            self.line_inference = LayoutDetection(self.platform, self.line_config)
            self.ready = True
        else:
            self.line_inference = None
            self.ready = False

    def update_ocr_model(self, config: OCRModelConfig):
        self.ocr_model_config = config
        self.ocr_inference = OCRInference(self.platform, config)

    def update_line_detection(self, config: Union[LineDetectionConfig, LayoutDetectionConfig]):
        if isinstance(config, LineDetectionConfig) and isinstance(self.line_config, LayoutDetectionConfig):
            self.line_inference = LineDetection(self.platform, config)
        elif isinstance(config, LayoutDetectionConfig) and isinstance(self.line_config, LineDetectionConfig):
            self.line_inference = LayoutDetection(self.platform, config)

        else:
            return


    # TODO: Generate specific meaningful error codes that can be returned inbetween the steps
    # TPS Mode is global-only at the moment
    def run_ocr(self,
                image: npt.NDArray,
                k_factor: float = 2.5,
                bbox_tolerance: float = 4.0,
                merge_lines: bool = True,
                use_tps: bool = False,
                tps_mode: TPSMode = TPSMode.GLOBAL,
                tps_threshold: float = 0.25,
                target_encoding: Encoding = Encoding.Unicode
                ):
        try:
            if not self.ready:
                return OpStatus.FAILED, "OCR pipeline not ready"

            if image is None:
                return OpStatus.FAILED, "Input image is None"

            # Get line mask
            try:
                if isinstance(self.line_config, LineDetectionConfig):
                    line_mask = self.line_inference.predict(image)
                else:
                    layout_mask = self.line_inference.predict(image)
                    line_mask = layout_mask[:, :, 2]
            except Exception as e:
                return OpStatus.FAILED, f"Line detection failed: {str(e)}"

            # Build line data
            try:
                rot_img, rot_mask, line_contours, page_angle = build_raw_line_data(image, line_mask)
                if len(line_contours) == 0:
                    return OpStatus.FAILED, "No lines detected"
            except Exception as e:
                return OpStatus.FAILED, f"Line data building failed: {str(e)}"

            # Filter contours
            filtered_contours = filter_line_contours(rot_mask, line_contours)
            if len(filtered_contours) == 0:
                return OpStatus.FAILED, "No valid lines after filtering"

            # Handle TPS (dewarping)
            try:
                if use_tps:
                    ratio, tps_line_data = check_for_tps(rot_img, filtered_contours)
                    if ratio > tps_threshold:
                        dewarped_img, dewarped_mask = apply_global_tps(rot_img, rot_mask, tps_line_data)
                        if len(dewarped_mask.shape) == 3:
                            dewarped_mask = cv2.cvtColor(dewarped_mask, cv2.COLOR_RGB2GRAY)
                        dew_rot_img, dew_rot_mask, line_contours, page_angle = build_raw_line_data(dewarped_img, dewarped_mask)
                        filtered_contours = filter_line_contours(dew_rot_mask, line_contours)
                        line_data = [build_line_data(x) for x in filtered_contours]
                        sorted_lines, _ = sort_lines_by_threshold2(rot_mask, line_data, group_lines=merge_lines)
                        line_images = extract_line_images(dew_rot_img, sorted_lines, k_factor, bbox_tolerance)
                    else:
                        line_data = [build_line_data(x) for x in filtered_contours]
                        sorted_lines, _ = sort_lines_by_threshold2(rot_mask, line_data, group_lines=merge_lines)
                        line_images = extract_line_images(rot_img, sorted_lines, k_factor, bbox_tolerance)
                else:
                    line_data = [build_line_data(x) for x in filtered_contours]
                    sorted_lines, _ = sort_lines_by_threshold2(rot_mask, line_data, group_lines=merge_lines)
                    line_images = extract_line_images(rot_img, sorted_lines, k_factor, bbox_tolerance)
            except Exception as e:
                return OpStatus.FAILED, f"Line processing failed: {str(e)}"

            # Process each line
            if line_images is not None and len(line_images) > 0:
                page_text = []
                ocr_lines = []
                try:
                    for line_img, line_info in zip(line_images, sorted_lines):
                        pred = self.ocr_inference.run(line_img)
                        pred = pred.strip()
                        pred = pred.replace("§", " ")

                        if self.encoder == CharsetEncoder.Wylie and target_encoding == Encoding.Unicode:
                            pred = self.converter.toUnicode(pred)
                        elif self.encoder == CharsetEncoder.Stack and target_encoding == Encoding.Wylie:
                            pred = self.converter.toWylie(pred)

                        ocr_line = OCRLine(
                            guid=line_info.guid,
                            text=pred,
                            encoding=Encoding.Wylie if target_encoding == Encoding.Wylie else Encoding.Unicode
                        )
                        ocr_lines.append(ocr_line)
                        page_text.append(pred)

                    return OpStatus.SUCCESS, (rot_mask, sorted_lines, ocr_lines, page_angle)
                except Exception as e:
                    return OpStatus.FAILED, f"OCR processing failed: {str(e)}"
            else:
                return OpStatus.FAILED, "No valid line images extracted"
        except Exception as e:
            return OpStatus.FAILED, f"OCR pipeline failed: {str(e)}"
