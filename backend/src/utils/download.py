import os
import requests
from pathlib import Path
import sys
# add parent directory to the path
sys.path.append(str(Path(__file__).parents[2].resolve()))
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pdf(url, output_folder, filename=None):
    """
    Downloads a PDF file from the given URL and saves it to the specified folder.

    Args:
        url (str): The URL of the PDF file.
        output_folder (str): The folder where the PDF will be saved.
        filename (str, optional): The name of the file. If None, the name is inferred from the URL.
    """
    try:
        # Ensure the output folder exists
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Infer filename if not provided
        if not filename:
            filename = url.split("/")[-1]

        # Full path to save the file
        file_path = os.path.join(output_folder, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Write the file to the output folder
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Downloaded {filename} to {output_folder}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    url = "https://cdn.cse.lk/cmt/upload_report_file/494_1717407552503.pdf"
    output_folder = Path(__file__).parents[2].resolve() / "data" / "raw"
    filename = "Haycarb_2024.pdf"
    download_pdf(url, output_folder, filename)