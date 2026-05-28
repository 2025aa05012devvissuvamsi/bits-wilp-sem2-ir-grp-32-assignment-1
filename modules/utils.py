"""
Utility functions for the
Information Retrieval system.

This module handles:
    1. Dataset visualization
    2. Text formatting
    3. Document preview rendering
    4. Preprocessing step display
"""

# ============================================================================
# IMPORTS
# ============================================================================

from typing import List

import pandas as pd
import streamlit as st
from pandas import DataFrame


# ============================================================================
# TEXT FORMATTING UTILITIES
# ============================================================================

def clean_document_text(
    text: str,
) -> str:
    """
    Clean extracted document text.

    This function:
        1. Removes excessive whitespace
        2. Removes repeated line breaks
        3. Normalizes spacing

    Args:
        text:
            Raw document text.

    Returns:
        str:
            Cleaned document text.
    """
    cleaned_text: str = " ".join(
        text.split()
    )

    return cleaned_text.strip()


def truncate_text(
    text: str,
    max_length: int = 250,
) -> str:
    """
    Truncate long text for UI display.

    Args:
        text:
            Input text.

        max_length:
            Maximum preview length.

    Returns:
        str:
            Truncated preview text.
    """
    if len(text) <= max_length:

        return text

    return (
        text[:max_length] + "..."
    )


# ============================================================================
# DATASET VISUALIZATION
# ============================================================================

def display_documents(
    dataframe: DataFrame,
) -> None:
    """
    Display uploaded document collection.

    This function:
        1. Cleans extracted text
        2. Creates readable previews
        3. Renders formatted dataframe

    Args:
        dataframe:
            Document collection dataframe.

    Returns:
        None
    """
    display_records: List[dict] = []

    for _, row in dataframe.iterrows():

        cleaned_text: str = (
            clean_document_text(
                text=str(row["text"])
            )
        )

        preview_text: str = truncate_text(
            text=cleaned_text,
            max_length=300,
        )

        display_records.append(
            {
                "Document ID": row["doc_id"],
                "Preview": preview_text,
            }
        )

    display_dataframe: DataFrame = (
        pd.DataFrame(display_records)
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
    )
