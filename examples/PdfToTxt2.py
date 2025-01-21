import pdfplumber
from tqdm import tqdm
import os
from pytesseract import image_to_string
from pdf2image import convert_from_path

# Get the list of available projects
base_folder = "./data anbud"
projects = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]

# Display available projects and prompt user to select one
print("Available projects:")
for i, project in enumerate(projects, 1):
    print(f"{i}. {project}")

project_index = int(input("Select the project number: ")) - 1
project = projects[project_index]

pdf_folder = f"{base_folder}/{project}/PDF"  # Update with the folder containing PDF files
text_path = f"{base_folder}/{project}/TXT/{project}.txt"  # Update with the actual output text file path

try:
    all_text = []

    # Iterate over all PDF files in the folder
    for filename in tqdm(os.listdir(pdf_folder), desc="Processing files", unit="file"):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # Open the PDF and extract text
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or "(cid:" in text:
                        # Use OCR as a fallback
                        print(f"OCR fallback for {filename}, page {page_number}")
                        page_image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)[0]
                        text = image_to_string(page_image)
                    all_text.append(text)

    # Join the extracted text with newlines
    combined_text = "\n".join(all_text)

    # Write the combined text to a file
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print("Text extraction from all PDFs completed successfully.")

except FileNotFoundError:
    print(f"Error: The folder {pdf_folder} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

