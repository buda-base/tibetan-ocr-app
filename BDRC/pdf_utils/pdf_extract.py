"""
Utilities for extracting embedded images from PDF files.
"""

import os

from pypdf import PdfReader


def extract_images_from_pdf(pdf_path, output_folder, first_page=1, last_page=None):
    """
    Extract the first image from each page of a PDF file using PdfReader.
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save extracted images
        first_page (int, optional): First page to process (1-based). Defaults to 1.
        last_page (int, optional): Last page to process. Defaults to None (all pages).
    Returns:
        list: List of paths to extracted images
        int: Total number of pages in the PDF
    """

    image_paths = []
    total_pages = 0
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if last_page is None or last_page > total_pages:
            last_page = total_pages
        if first_page < 1:
            first_page = 1
        first_page_idx = first_page - 1
        last_page_idx = last_page - 1
        os.makedirs(output_folder, exist_ok=True)
        file_n = os.path.splitext(os.path.basename(pdf_path))[0]
        for idx in range(first_page_idx, last_page_idx + 1):
            page = reader.pages[idx]
            if hasattr(page, "images") and len(page.images) > 0:
                img_obj = page.images[0]
                data = img_obj.data
                tmp_img_path = os.path.join(output_folder, f"{file_n} - page {idx+1}.jpg")
                with open(tmp_img_path, "wb") as f:
                    f.write(data)
                image_paths.append(tmp_img_path)
    except Exception as e:
        raise RuntimeError(f"Error extracting images from PDF: {e}")
    return image_paths, total_pages
