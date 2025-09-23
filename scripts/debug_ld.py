import os
import sys
import argparse
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BDRC.Utils import get_platform
from BDRC.Data import LineDetectionConfig, LayoutDetectionConfig
from BDRC.Inference import LineDetection, LayoutDetection
from BDRC.line_detection import (
    build_raw_line_data,
    filter_line_contours,
    build_line_data,
    sort_lines_by_threshold2,
    extract_line_images
)
from BDRC.image_dewarping import (
    check_for_tps,
    apply_global_tps
)


def main():
    parser = argparse.ArgumentParser(description="Debug line detection process.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--k-factor", type=float, default=2.5, help="Line extraction k-factor")
    parser.add_argument("--bbox-tolerance", type=float, default=3.0, help="Bounding box tolerance")
    parser.add_argument("--dewarp", action="store_true", help="Apply TPS dewarping")
    parser.add_argument("--line-mode", choices=["line", "layout"], default="line", help="Line detection mode")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image at {args.image_path}")
        sys.exit(1)

    # Create output directory
    image_filename = os.path.basename(args.image_path)
    image_name, image_ext = os.path.splitext(image_filename)
    output_dir = os.path.join("debug_ld_output", f"{image_name}_{image_ext.replace('.', '')}")
    os.makedirs(output_dir, exist_ok=True)

    # Get platform
    platform = get_platform()

    # Select line detection config
    if args.line_mode == "line":
        line_config = LineDetectionConfig(model_file="../Models/Lines/PhotiLines.onnx", patch_size=512)
        line_inference = LineDetection(platform, line_config)
    else:
        line_config = LayoutDetectionConfig(model_file="../Models/Layout/photi.onnx", patch_size=512, classes=["background", "image", "line", "caption", "margin"])
        line_inference = LayoutDetection(platform, line_config)

    # Line detection
    if isinstance(line_config, LineDetectionConfig):
        line_mask = line_inference.predict(image)
    else:
        layout_mask = line_inference.predict(image)
        line_mask = layout_mask[:, :, 2]
    
    # Build line data
    rot_img, rot_mask, line_contours, page_angle = build_raw_line_data(image, line_mask)
    
    # Filter contours
    filtered_contours = filter_line_contours(rot_mask, line_contours)

    # Handle TPS (dewarping)
    dewarped_img = None
    if args.dewarp:
        ratio, tps_line_data = check_for_tps(rot_img, filtered_contours)
        if ratio > 0.25: # threshold from OCRPipeline
            dewarped_img, dewarped_mask = apply_global_tps(rot_img, rot_mask, tps_line_data)
            if len(dewarped_mask.shape) == 3:
                dewarped_mask = cv2.cvtColor(dewarped_mask, cv2.COLOR_RGB2GRAY)
            dew_rot_img, dew_rot_mask, line_contours, page_angle = build_raw_line_data(dewarped_img, dewarped_mask)
            filtered_contours = filter_line_contours(dew_rot_mask, line_contours)
            rot_img = dew_rot_img
            rot_mask = dew_rot_mask

    # --- Visualization ---

    # 1. Output of the line detection model
    line_detection_output_img = create_colored_overlay(image, line_mask, (0, 255, 0))
    cv2.imwrite(os.path.join(output_dir, "01_line_detection_output.png"), line_detection_output_img)

    # 2. Segmented lines, before expansion
    segmented_before_expansion = np.zeros_like(rot_img)
    cv2.drawContours(segmented_before_expansion, filtered_contours, -1, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "02_segmented_lines_before_expansion.png"), segmented_before_expansion)

    # 3. Segmented lines, after expansion
    line_data = [build_line_data(x) for x in filtered_contours]
    sorted_lines, _ = sort_lines_by_threshold2(rot_mask, line_data, group_lines=True)
    segmented_after_expansion = np.zeros_like(rot_img)
    cv2.drawContours(segmented_after_expansion, [l.contour for l in sorted_lines], -1, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "03_segmented_lines_after_expansion.png"), segmented_after_expansion)

    # 4. Dewarped image
    if dewarped_img is not None:
        cv2.imwrite(os.path.join(output_dir, "06_dewarped_image.png"), dewarped_img)

        # 5. Dewarped segmented lines, before and after expansion
        # Re-run line detection on dewarped image to get correct contours
        if isinstance(line_config, LineDetectionConfig):
            dewarped_line_mask = line_inference.predict(dewarped_img)
        else:
            dewarped_layout_mask = line_inference.predict(dewarped_img)
            dewarped_line_mask = dewarped_layout_mask[:, :, 2]

        dew_rot_img, dew_rot_mask, dew_line_contours, _ = build_raw_line_data(dewarped_img, dewarped_line_mask)
        dew_filtered_contours = filter_line_contours(dew_rot_mask, dew_line_contours)
        
        dewarped_segmented_before = np.zeros_like(dew_rot_img)
        cv2.drawContours(dewarped_segmented_before, dew_filtered_contours, -1, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "04_dewarped_segmented_lines_before_expansion.png"), dewarped_segmented_before)

        dew_line_data = [build_line_data(x) for x in dew_filtered_contours]
        dew_sorted_lines, _ = sort_lines_by_threshold2(dew_rot_mask, dew_line_data, group_lines=True)
        dewarped_segmented_after = np.zeros_like(dew_rot_img)
        cv2.drawContours(dewarped_segmented_after, [l.contour for l in dew_sorted_lines], -1, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "05_dewarped_segmented_lines_after_expansion.png"), dewarped_segmented_after)

    print(f"Debug images saved to {output_dir}")


def create_colored_overlay(image, mask, color, alpha=0.5):
    """Creates a colored overlay on an image from a mask."""
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


if __name__ == "__main__":
    main()
