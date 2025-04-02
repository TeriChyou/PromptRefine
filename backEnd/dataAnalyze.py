import os
from PyPDF2 import PdfReader
# custom modules
import functions as func

# 處理檔案
def extract_file_content(file_path):
    extension = os.path.splitext(file_path)[-1].lower()

    try:
        if extension == '.pdf':
            reader = PdfReader(file_path)
            text = ''.join([page.extract_text() for page in reader.pages])
            return text

        else:
            return f"Unsupported file type: {extension}"

    except Exception as e:
        return f"Error processing file: {e}"

