import argparse
import glob
import os
import sys

import cv2

from BDRC.artifact_manager import ArtifactManager
from BDRC.audit_logger import AuditLogger
from BDRC.data import ArtifactConfig, Encoding, LayoutDetectionConfig, LineDetectionConfig
from BDRC.exporter import TextExporter
from BDRC.inference import OCRPipeline
from BDRC.pipeline import run_ocr_with_artifacts
from BDRC.utils import get_platform, import_local_model

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


def main():
    parser = argparse.ArgumentParser(description="Run Tibetan OCR inference on images.")
    parser.add_argument("--model", required=True, help="Path to OCR model directory")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--folder", help="Path to a folder containing images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--encoding", choices=["unicode", "wylie"], default="unicode", help="Output encoding")
    parser.add_argument("--k-factor", type=float, default=2.5, help="Line extraction k-factor")
    parser.add_argument("--bbox-tolerance", type=float, default=4.0, help="Bounding box tolerance")
    parser.add_argument("--merge-lines", action="store_true", help="Merge line chunks")
    parser.add_argument("--dewarp", action="store_true", help="Apply TPS dewarping")
    parser.add_argument("--line-mode", choices=["line", "layout"], default="line", help="Line detection mode")
    parser.add_argument("--save-artifacts", action="store_true", help="Enable artifact saving")
    parser.add_argument("--artifact-output", default="output", help="Base directory for artifacts")
    parser.add_argument(
        "--artifact-granularity",
        choices=["minimal", "standard"],
        default="standard",
        help="Level of artifact detail to save",
    )
    args = parser.parse_args()

    if args.image and args.folder:
        parser.error("--image and --folder cannot be used together.")
    if not args.image and not args.folder:
        parser.error("You must specify either --image or --folder.")

    os.makedirs(args.output, exist_ok=True)

    # Load model
    config_path = os.path.join(args.model, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    ocr_model = import_local_model(os.path.dirname(args.model))

    # Line detection config
    if args.line_mode == "line":
        line_config = LineDetectionConfig(model_file="Models/Lines/PhotiLines.onnx", patch_size=512)
    else:
        line_config = LayoutDetectionConfig(
            model_file="Models/Layout/photi.onnx",
            patch_size=512,
            classes=["background", "image", "line", "caption", "margin"],
        )

    pipeline = OCRPipeline(get_platform(), ocr_model.config, line_config)
    target_encoding = Encoding.UNICODE if args.encoding == "unicode" else Encoding.WYLIE

    # Collect images
    is_batch_mode = bool(args.folder)
    if args.folder:
        image_paths = [p for ext in IMAGE_EXTENSIONS for p in glob.glob(os.path.join(args.folder, ext))]
        if not image_paths:
            print(f"No images found in {args.folder}")
            sys.exit(1)
    else:
        image_paths = [args.image]

    # Artifact setup
    artifact_manager = None
    audit_logger = None
    artifact_config = None

    if args.save_artifacts:
        is_standard = args.artifact_granularity == "standard"
        artifact_config = ArtifactConfig(
            enabled=True, granularity=args.artifact_granularity, save_detection=is_standard, save_dewarping=is_standard
        )

        artifact_manager = ArtifactManager(
            base_output_dir=args.artifact_output,
            job_id=None,
            config={
                "image_count": len(image_paths),
                "image_paths": [os.path.basename(p) for p in image_paths],
                "k_factor": args.k_factor,
                "bbox_tolerance": args.bbox_tolerance,
                "merge_lines": args.merge_lines,
                "dewarp": args.dewarp,
                "encoding": args.encoding,
                "line_mode": args.line_mode,
                "artifact_granularity": args.artifact_granularity,
            },
        )
        artifact_manager.create_directory_structure()
        artifact_manager.save_config()

        if is_standard:
            audit_logger = AuditLogger(artifact_manager.job_id, artifact_manager.job_dir / "audit.log")

    # Process images
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            if audit_logger:
                audit_logger.log_error(f"Failed to load image: {img_path}")
            continue

        page_name = os.path.basename(img_path)
        base = os.path.splitext(page_name)[0]

        if artifact_manager and is_batch_mode:
            artifact_manager.set_current_page(page_name)

        status, result = run_ocr_with_artifacts(
            pipeline=pipeline,
            image=img,
            image_name=base,
            k_factor=args.k_factor,
            bbox_tolerance=args.bbox_tolerance,
            merge_lines=args.merge_lines,
            use_tps=args.dewarp,
            target_encoding=target_encoding,
            artifact_manager=artifact_manager,
            audit_logger=audit_logger,
            artifact_config=artifact_config,
        )

        if status.name == "SUCCESS":
            _, lines, ocr_lines, angle = result
            if not artifact_manager:
                TextExporter(args.output).export_lines(img, base, lines, ocr_lines, angle=angle)
                print(f"Text output: {args.output}/{base}.txt")
        else:
            print(f"OCR failed for {img_path}: {result}")
            if audit_logger:
                audit_logger.log_error(f"Pipeline failed for {page_name}: {result}")

    # Finalize
    if artifact_manager:
        if is_batch_mode:
            artifact_manager.save_aggregate_metrics()
        artifact_manager.generate_manifest()
        print(f"Artifacts saved to: {artifact_manager.job_dir}")
        if audit_logger:
            print(f"Audit log available at: {artifact_manager.job_dir / 'audit.log'}")


if __name__ == "__main__":
    main()
