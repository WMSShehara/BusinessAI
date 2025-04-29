import os
import sys
from pathlib import Path
import pdfplumber
import pandas as pd
import io
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Add parent directory to the path
sys.path.append(str(Path(__file__).parents[2].resolve()))
from src.lib.logger import get_logger

# Set up logging
logger = get_logger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path, output_path):
        self.pdf_path = pdf_path
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.pdf = None
        self.metadata = {}

    def open_pdf(self):
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
            self.metadata = self.pdf.metadata
            logger.info(f"PDF opened successfully: {self.pdf_path}")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    def close_pdf(self):
        if self.pdf:
            self.pdf.close()
            logger.info("PDF closed.")

    def extract_tables(self, page):
        """Extracts tables from a PDF page."""
        tables = page.extract_tables()
        return tables

    def save_table_as_csv(self, table, table_index, page_number):
        """Saves an extracted table to a CSV file."""
        try:
            df = pd.DataFrame(table)
            csv_filename = os.path.join(self.output_path, f"page_{page_number}_table_{table_index}.csv")
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            logger.info(f"Table saved to {csv_filename}")
        except Exception as e:
            logger.error(f"Failed to save table to CSV: {e}")

    def extract_figures(self, page):
        """Extracts figures (images) from a PDF page."""
        images = []
        for obj in page.objects.values():
            if 'stream' in obj:
                stream = obj['stream']
                if 'Type' in stream and stream['Type'] == 'XObject' and 'Subtype' in stream and stream['Subtype'] == 'Image':
                    try:
                        img = Image.open(io.BytesIO(stream.rawdata))
                        images.append(img)
                    except Exception as e:
                        logger.error(f"Error extracting image: {e}")
        return images

    def save_figure(self, image, figure_index, page_number):
         """Saves an extracted image to a file."""
         try:
             image_filename = os.path.join(self.output_path, f"page_{page_number}_figure_{figure_index}.png")
             image.save(image_filename, "PNG")
             logger.info(f"Figure saved to {image_filename}")
         except Exception as e:
             logger.error(f"Failed to save figure: {e}")

    def extract_text_with_layout(self, page):
        """Extracts text from a PDF page, preserving layout information."""
        blocks = page.extract_words()
        return blocks

    def chunk_text_by_pages(self):
        """Chunks text by pages."""
        chunks = []
        for i, page in enumerate(self.pdf.pages):
            page_number = i + 1
            text = page.extract_text()
            if text:
                chunks.append((page_number, text))
        return chunks

    def remove_headers_and_footers(self, page):
        """Removes headers and footers from a PDF page."""
        text = page.extract_text()
        if not text:
            return ""

        lines = text.split("\n")
        if len(lines) > 2:
            # Remove the first and last lines as headers and footers
            lines = lines[1:-1]
        return "\n".join(lines)

    def save_combined_chunks(self):
        """Combines all text chunks into a single Markdown file, removing headers and footers."""
        combined_filename = os.path.join(self.output_path, "combined_output.md")
        with open(combined_filename, "w", encoding="utf-8") as combined_file:
            for i, page in enumerate(self.pdf.pages):
                page_number = i + 1
                text = self.remove_headers_and_footers(page)
                if text:
                    combined_file.write(f"# Page {page_number}\n\n")
                    combined_file.write(text + "\n\n")
        logger.info(f"Combined output saved to {combined_filename}")

    def process_pdf(self):
        """Processes the PDF to extract text by pages and save combined output."""
        self.open_pdf()
        if not self.pdf:
            return

        # Save combined text chunks into a single Markdown file
        self.save_combined_chunks()

        self.close_pdf()

if __name__ == "__main__":
    # Example usage
    pdf_path = Path(__file__).parents[2].resolve() / "data" / "raw" / "Haycarb_2024.pdf"
    output_path = Path(__file__).parents[2].resolve() / "data" / "output"/"Processed_Haycarb_2024"
    processor = PDFProcessor(pdf_path, output_path)
    processor.process_pdf()