"""OCR Pipeline wrapper with artifact management and audit logging."""

import time
from typing import Any, Optional, Tuple

import numpy.typing as npt

from BDRC.artifact_manager import ArtifactManager
from BDRC.audit_logger import AuditLogger
from BDRC.data import ArtifactConfig, Encoding, Line, OpStatus
from BDRC.exporter import PageXMLExporter, TextExporter
from BDRC.inference import OCRPipeline


def serialize_contours(contours) -> list:
    return [c.tolist() for c in contours]


def serialize_lines(lines: list[Line]) -> list:
    return [
        {
            "guid": str(ln.guid),
            "bbox": {"x": ln.bbox.x, "y": ln.bbox.y, "w": ln.bbox.w, "h": ln.bbox.h},
            "center": ln.center,
        }
        for ln in lines
    ]


def run_ocr_with_artifacts(
    pipeline: OCRPipeline,
    image: npt.NDArray,
    image_name: str,
    k_factor: float = 2.5,
    bbox_tolerance: float = 4.0,
    merge_lines: bool = True,
    use_tps: bool = False,
    tps_threshold: float = 0.25,
    target_encoding: Encoding = Encoding.UNICODE,
    artifact_manager: Optional[ArtifactManager] = None,
    audit_logger: Optional[AuditLogger] = None,
    artifact_config: Optional[ArtifactConfig] = None,
) -> Tuple[OpStatus, Any]:
    """Run OCR pipeline with artifact saving and audit logging."""

    pipeline_start = time.perf_counter()
    save_det = artifact_manager and artifact_config and artifact_config.save_detection
    save_dew = artifact_manager and artifact_config and artifact_config.save_dewarping

    def log_start(stage, meta=None):
        if audit_logger:
            audit_logger.log_stage_start(stage, metadata=meta)

    def log_end(stage, meta=None):
        if audit_logger:
            audit_logger.log_stage_end(stage, status="success", metadata=meta)

    def log_err(msg, stage):
        if audit_logger:
            audit_logger.log_error(msg, stage=stage)

    log_start(
        "ocr_pipeline",
        {
            "image_name": image_name,
            "image_shape": image.shape,
            "k_factor": k_factor,
            "bbox_tolerance": bbox_tolerance,
            "merge_lines": merge_lines,
            "use_tps": use_tps,
            "target_encoding": str(target_encoding),
        },
    )

    if artifact_manager:
        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

    try:
        # STAGE 1: Line/Layout Detection
        log_start("line_detection")
        status, line_mask = pipeline.detect_lines(image)
        if status == OpStatus.FAILED:
            log_err(line_mask, "line_detection")
            return status, line_mask
        if save_det:
            artifact_manager.save_image("line_mask", line_mask, "detection")
        log_end("line_detection", {"mask_shape": line_mask.shape})

        # STAGE 2: Build Line Data
        log_start("build_line_data")
        status, result = pipeline.build_lines(image, line_mask)
        if status == OpStatus.FAILED:
            log_err(result, "build_line_data")
            return status, result
        rot_img, rot_mask, line_contours, filtered_contours, page_angle = result
        if save_det:
            artifact_manager.save_image("rotated_mask", rot_mask, "detection")
            artifact_manager.save_json(
                "contours_raw",
                {"count": len(line_contours), "contours": serialize_contours(line_contours)},
                "detection",
            )
            artifact_manager.save_json(
                "contours_filtered",
                {"count": len(filtered_contours), "contours": serialize_contours(filtered_contours)},
                "detection",
            )
        log_end(
            "build_line_data",
            {
                "rotation_angle": page_angle,
                "contour_count": len(line_contours),
                "filtered_count": len(filtered_contours),
            },
        )

        # STAGE 3: TPS Dewarping
        log_start("dewarping")
        status, dewarp_result = pipeline.apply_dewarping(
            rot_img, rot_mask, filtered_contours, page_angle, use_tps=use_tps, tps_threshold=tps_threshold
        )
        if status == OpStatus.FAILED:
            log_err(dewarp_result, "dewarping")
            return status, dewarp_result
        if save_dew and dewarp_result.tps_ratio is not None:
            artifact_manager.save_json(
                "tps_analysis",
                {"ratio": float(dewarp_result.tps_ratio), "threshold": tps_threshold, "applied": dewarp_result.applied},
                "dewarping",
            )
            if dewarp_result.applied and dewarp_result.dewarped_mask is not None:
                artifact_manager.save_image("dewarped_mask", dewarp_result.dewarped_mask, "dewarping")
        log_end("dewarping", {"tps_ratio": dewarp_result.tps_ratio, "dewarping_applied": dewarp_result.applied})

        # STAGE 4: Extract Lines
        log_start("extract_lines")
        status, result = pipeline.extract_lines(
            dewarp_result.work_img,
            rot_mask,
            dewarp_result.filtered_contours,
            merge_lines=merge_lines,
            k_factor=k_factor,
            bbox_tolerance=bbox_tolerance,
        )
        if status == OpStatus.FAILED:
            log_err(result, "extract_lines")
            return status, result
        sorted_lines, line_images = result
        if artifact_manager and artifact_config:
            artifact_manager.save_json(
                "lines", {"count": len(sorted_lines), "lines": serialize_lines(sorted_lines)}, "lines"
            )
        log_end("extract_lines", {"lines_extracted": len(sorted_lines)})

        # STAGE 5: OCR Inference
        log_start("ocr_inference")
        status, ocr_lines = pipeline.run_text_recognition(line_images, sorted_lines, target_encoding=target_encoding)
        if status == OpStatus.FAILED:
            log_err(ocr_lines, "ocr_inference")
            return status, ocr_lines
        if audit_logger:
            for idx in range(len(ocr_lines)):
                audit_logger.log_operation(f"ocr_line_{idx+1}", stage="ocr_inference")
        log_end("ocr_inference", {"lines_processed": len(ocr_lines)})

        # STAGE 6: Save Results
        if artifact_manager:
            results_dir = artifact_manager.get_results_dir()
            TextExporter(str(results_dir)).export_lines(image, image_name, sorted_lines, ocr_lines)
            PageXMLExporter(str(results_dir)).export_lines(image, image_name, sorted_lines, ocr_lines, angle=page_angle)

        # Pipeline Complete
        pipeline_duration = (time.perf_counter() - pipeline_start) * 1000
        log_end("ocr_pipeline")

        if artifact_manager:
            artifact_manager.save_metrics(
                {
                    "total_duration_ms": pipeline_duration,
                    "lines_detected": len(sorted_lines),
                    "lines_processed": len(ocr_lines),
                    "dewarping_applied": dewarp_result.applied,
                    "rotation_angle": page_angle,
                    "image_name": image_name,
                }
            )

        return OpStatus.SUCCESS, (rot_mask, sorted_lines, ocr_lines, page_angle)

    except Exception as e:
        log_err(f"OCR pipeline failed: {e}", "ocr_pipeline")
        if audit_logger:
            audit_logger.log_stage_end("ocr_pipeline", status="failure")
        return OpStatus.FAILED, f"OCR pipeline failed: {e}"
