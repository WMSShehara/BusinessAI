import layoutparser as lp
import pytesseract
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import json
import logging
import gc
import time


class DocumentProcessor:
    """Processes a PDF into structured layout-aware chunks using LayoutParser and Tesseract OCR."""

    def __init__(self, publaynet_thresh: float = 0.5):
        """Initialize the DocumentProcessor."""
        self.publaynet_thresh = publaynet_thresh
        self._model = None

    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            try:
                logging.info("Starting layout model initialization...")
                start_time = time.time()

                self._model = lp.AutoLayoutModel(
                    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                    label_map={
                        0: "Text",
                        1: "Title",
                        2: "List",
                        3: "Table",
                        4: "Figure",
                    },
                    extra_config={
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST": self.publaynet_thresh
                    },
                )

                end_time = time.time()
                logging.info(
                    f"Successfully initialized layout model in {end_time - start_time:.2f} seconds"
                )
            except Exception as e:
                logging.warning(f"Layout model initialization failed: {e}")
                logging.info("Will process documents without layout detection")
                self._model = None
        return self._model

    def process_pdf(
        self, pdf_path: str, save_json: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF and return structured chunks. Optionally save to JSON.
        Args:
            pdf_path: Path to the PDF file.
            save_json: If provided, save the output to this JSON file.
        Returns:
            List of structured chunk dicts.
        """
        try:
            info = pdfinfo_from_path(pdf_path)
            num_pages = info["Pages"]
            all_chunks = []

            for page_num in range(1, num_pages + 1):
                try:
                    # Process one page at a time, low DPI
                    pil_imgs = convert_from_path(
                        pdf_path, dpi=100, first_page=page_num, last_page=page_num
                    )
                    if not pil_imgs:
                        logging.warning(f"No images extracted from page {page_num}")
                        continue

                    pil_img = pil_imgs[0]
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    if self.model is not None:
                        # Use layout detection if model is available
                        layout = self.model.detect(img)
                        if layout is not None:
                            layout = sorted(
                                layout, key=lambda b: (b.block.y_1, b.block.x_1)
                            )
                            self._process_layout_blocks(
                                layout, img, page_num, all_chunks
                            )
                        else:
                            # Fallback to full page processing if layout detection fails
                            self._process_full_page(img, page_num, all_chunks)
                    else:
                        # Process without layout detection
                        self._process_full_page(img, page_num, all_chunks)

                    # Clean up memory
                    del pil_img, img, pil_imgs
                    gc.collect()

                except Exception as e:
                    logging.error(f"Error processing page {page_num}: {e}")
                    continue

            if save_json:
                with open(save_json, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

            return all_chunks

        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def _process_layout_blocks(self, layout, img, page_num, all_chunks):
        """Process blocks detected by layout model."""
        section = None
        for block in layout:
            try:
                x1, y1, x2, y2 = map(int, block.block.points[0] + block.block.points[2])
                crop = img[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                text = ""
                if block.type in ["Title", "Text", "List"]:
                    text = pytesseract.image_to_string(pil_crop, lang="eng").strip()

                chunk_type = (
                    "section"
                    if block.type == "Title"
                    else "list"
                    if block.type == "List"
                    else "paragraph"
                    if block.type == "Text"
                    else "visual"
                    if block.type == "Figure"
                    else "table"
                    if block.type == "Table"
                    else "other"
                )

                if block.type == "Title" and text:
                    section = text

                # Convert bbox list to string
                bbox_str = f"{x1},{y1},{x2},{y2}"

                chunk = {
                    "type": chunk_type,
                    "section": section if section else "",
                    "text": text if chunk_type != "visual" else "",
                    "page": page_num,
                    "bbox": bbox_str,  # Store as string instead of list
                }
                all_chunks.append(chunk)

            except Exception as e:
                logging.error(f"Error processing block: {e}")
                continue

    def _process_full_page(self, img, page_num, all_chunks):
        """Process entire page when layout detection is not available."""
        try:
            # Convert to PIL Image for OCR
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, lang="eng").strip()

            if text:
                # Convert bbox list to string
                bbox_str = f"0,0,{img.shape[1]},{img.shape[0]}"

                chunk = {
                    "type": "paragraph",
                    "section": "",
                    "text": text,
                    "page": page_num,
                    "bbox": bbox_str,  # Store as string instead of list
                }
                all_chunks.append(chunk)
        except Exception as e:
            logging.error(f"Error processing full page: {e}")

    def process_all_documents(self, input_dir: Path, output_dir: Path) -> list[dict]:
        """
        Process all PDF files in the input directory and save structured JSONs to output directory.

        Args:
            input_dir (Path): Directory containing PDF files.
            output_dir (Path): Directory to save structured JSON files.

        Returns:
            List[dict]: All extracted chunks from all PDFs.
        """
        all_chunks = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in input_dir.glob("*.pdf"):
            output_json = output_dir / f"{pdf_path.stem}_structured.json"
            try:
                chunks = self.process_pdf(str(pdf_path), save_json=str(output_json))
                all_chunks.extend(chunks)
                logging.info(
                    f"Processed {pdf_path} -> {output_json} ({len(chunks)} chunks)"
                )
            except Exception as e:
                logging.error(f"Failed to process {pdf_path}: {e}")
        return all_chunks
