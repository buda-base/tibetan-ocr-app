import os
import sys
import argparse
import glob
import cv2
from BDRC.Utils import import_local_model, get_platform
from BDRC.Data import Encoding, LineMode, OCRSettings
from BDRC.Inference import OCRPipeline
from BDRC.Data import LineDetectionConfig, LayoutDetectionConfig, OCRModelConfig

def find_model_config(model_dir):
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found in {model_dir}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Run Tibetan OCR inference on images.")
    parser.add_argument("--model", required=True, help="Path to OCR model directory (must contain model_config.json)")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--folder", help="Path to a folder containing images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--encoding", choices=["unicode", "wylie"], default="unicode", help="Output encoding")
    parser.add_argument("--k-factor", type=float, default=2.5, help="Line extraction k-factor")
    parser.add_argument("--bbox-tolerance", type=float, default=4.0, help="Bounding box tolerance")
    parser.add_argument("--merge-lines", action="store_true", help="Merge line chunks")
    parser.add_argument("--dewarp", action="store_true", help="Apply TPS dewarping")
    parser.add_argument("--line-mode", choices=["line", "layout"], default="line", help="Line detection mode")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # Load model config
    model_dir = args.model
    config_path = find_model_config(model_dir)
    ocr_model = import_local_model(os.path.dirname(model_dir))

    # Select line detection config
    if args.line_mode == "line":
        # Dummy config, you may want to allow this as argument
        line_config = LineDetectionConfig(model_file="Models/Lines/PhotiLines.onnx", patch_size=512)
    else:
        # Dummy config, you may want to allow this as argument
        line_config = LayoutDetectionConfig(model_file="Models/Layout/photi.onnx", patch_size=512, classes=["background", "image", "line", "caption", "margin"])

    platform = get_platform()
    pipeline = OCRPipeline(platform, ocr_model.config, line_config)

    # Prepare image list
    if args.image:
        image_paths = [args.image]
    elif args.folder:
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
        if not image_paths:
            print(f"No images found in {args.folder}")
            sys.exit(1)
    else:
        print("You must specify either --image or --folder")
        sys.exit(1)

    # Run inference
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        status, result = pipeline.run_ocr(
            image=img,
            k_factor=args.k_factor,
            bbox_tolerance=args.bbox_tolerance,
            merge_lines=args.merge_lines,
            use_tps=args.dewarp,
            target_encoding=Encoding.Unicode if args.encoding == "unicode" else Encoding.Wylie
        )
        if status.name == "SUCCESS":
            rot_mask, lines, ocr_lines, angle = result
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_txt = os.path.join(args.output, base + ".txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                for line in ocr_lines:
                    f.write(line.text + "\n")
            print(f"OCR for {img_path} written to {out_txt}")
        else:
            print(f"OCR failed for {img_path}: {result}")

if __name__ == "__main__":
    main()
