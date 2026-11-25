"""
text_extraction.py
-------------------
Handles extracting raw text from resume files (PDF, DOCX, TXT).

Supported formats:
    - .pdf  (via pdfplumber, falls back to PyPDF2)
    - .docx (via python-docx)
    - .txt  (plain read)
"""

import os
import io


def extract_text_from_pdf(file_obj_or_path):
    """
    Extract text from a PDF file.
    Accepts either a file path (str) or a file-like object (e.g. from Streamlit uploader).
    """
    text_parts = []
    try:
        import pdfplumber
        with pdfplumber.open(file_obj_or_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception:
        # Fallback to PyPDF2 if pdfplumber fails or is unavailable
        try:
            from PyPDF2 import PdfReader
            if hasattr(file_obj_or_path, "read"):
                reader = PdfReader(file_obj_or_path)
            else:
                reader = PdfReader(open(file_obj_or_path, "rb"))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(file_obj_or_path):
    """
    Extract text from a DOCX file, including paragraphs and tables.
    """
    try:
        import docx
        document = docx.Document(file_obj_or_path)

        text_parts = [p.text for p in document.paragraphs if p.text.strip()]

        # Also pull text out of any tables (some resumes use table layouts)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)

        return "\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from DOCX: {e}")


def extract_text_from_txt(file_obj_or_path):
    """
    Extract text from a plain .txt file.
    """
    if hasattr(file_obj_or_path, "read"):
        raw = file_obj_or_path.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return raw
    with open(file_obj_or_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_obj, filename=None):
    """
    Dispatcher function: detects file type by extension and routes to the
    appropriate extractor.

    Parameters
    ----------
    file_obj : file-like object or path (str)
    filename : str, optional
        Original filename (needed when file_obj is an in-memory buffer,
        e.g. a Streamlit UploadedFile, to detect extension).

    Returns
    -------
    str : extracted plain text
    """
    if filename is None:
        filename = getattr(file_obj, "name", None) or (file_obj if isinstance(file_obj, str) else "")

    ext = os.path.splitext(filename)[1].lower()

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if ext == ".pdf":
        return extract_text_from_pdf(file_obj)
    elif ext == ".docx":
        return extract_text_from_docx(file_obj)
    elif ext == ".txt":
        return extract_text_from_txt(file_obj)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please upload PDF, DOCX, or TXT.")
