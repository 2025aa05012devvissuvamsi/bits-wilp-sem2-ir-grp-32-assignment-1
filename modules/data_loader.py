"""
Data loading utilities for the
Information Retrieval system.

This module handles:
    1. Default dataset loading
    2. Uploaded CSV dataset loading
    3. PDF document ingestion
    4. Dataset schema validation
    5. Dataframe generation

Supported input formats:
    - CSV
    - PDF
"""

# ============================================================================
# IMPORTS
# ============================================================================

from io import BytesIO
from typing import Dict
from typing import List

import fitz
import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import (
    UploadedFile,
)


# ============================================================================
# DEFAULT DATASET LOADING
# ============================================================================

# ============================================================================
# DEFAULT DATASET LOADING
# ============================================================================

def load_default_dataset() -> DataFrame:
    """
    Load default sample dataset from CSV file.

    Returns:
        DataFrame:
            Default document collection.
    """
    return pd.read_csv(
        "data/sample_default.csv"
    )

# ============================================================================
# CSV DATASET LOADING
# ============================================================================

def load_uploaded_dataset(
    uploaded_file: UploadedFile,
) -> DataFrame:
    """
    Load uploaded CSV dataset.

    Expected schema:
        - doc_id
        - text

    Args:
        uploaded_file:
            Uploaded CSV dataset file.

    Returns:
        DataFrame:
            Loaded document collection.

    Raises:
        ValueError:
            If required columns are missing.
    """
    documents_df: DataFrame = pd.read_csv(
        uploaded_file
    )

    required_columns: List[str] = [
        "doc_id",
        "text",
    ]

    missing_columns: List[str] = [
        column
        for column in required_columns
        if column not in documents_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Uploaded dataset is missing "
            f"required columns: {missing_columns}"
        )

    return documents_df


# ============================================================================
# PDF DOCUMENT INGESTION
# ============================================================================

def extract_text_from_pdf(
    uploaded_pdf: UploadedFile,
) -> str:
    """
    Extract textual content from uploaded PDF.

    Args:
        uploaded_pdf:
            Uploaded PDF file.

    Returns:
        str:
            Extracted PDF text.
    """
    pdf_stream: BytesIO = BytesIO(
        uploaded_pdf.read()
    )

    pdf_document = fitz.open(
        stream=pdf_stream.read(),
        filetype="pdf",
    )

    extracted_text: str = ""

    for page in pdf_document:

        extracted_text += page.get_text()

    pdf_document.close()

    return extracted_text.strip()


def load_pdf_documents(
    uploaded_pdf_files: List[UploadedFile],
) -> DataFrame:
    """
    Load and process uploaded PDF documents.

    Workflow:
        1. Read uploaded PDF files
        2. Extract text from pages
        3. Generate document records
        4. Create dataframe

    Args:
        uploaded_pdf_files:
            Uploaded PDF collection.

    Returns:
        DataFrame:
            Extracted PDF document dataframe.
    """
    document_records: List[
        Dict[str, object]
    ] = []

    document_counter: int = 1

    for uploaded_pdf in uploaded_pdf_files:

        extracted_text: str = (
            extract_text_from_pdf(
                uploaded_pdf=uploaded_pdf
            )
        )

        document_records.append(
            {
                "doc_id": document_counter,
                "text": extracted_text,
            }
        )

        document_counter += 1

    return pd.DataFrame(
        document_records
    )