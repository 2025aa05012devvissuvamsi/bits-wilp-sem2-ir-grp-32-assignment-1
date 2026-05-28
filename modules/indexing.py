"""
Indexing module for Information Retrieval system.

This module contains utilities for creating
inverted indexes from preprocessed documents.
"""

from collections import defaultdict
from typing import Dict, List


def create_inverted_index(
    processed_documents: Dict[int, List[str]]
) -> Dict[str, List[int]]:
    """
    Create an inverted index from processed documents.

    Inverted index structure:
        term -> list of document IDs

    Args:
        processed_documents:
            Dictionary mapping document ID
            to processed token sequence.

    Returns:
        Dict[str, List[int]]:
            Generated inverted index.
    """
    inverted_index: Dict[str, List[int]] = defaultdict(list)

    for document_id, tokens in processed_documents.items():

        unique_tokens = set(tokens)

        for token in unique_tokens:

            if document_id not in inverted_index[token]:
                inverted_index[token].append(document_id)

    return dict(inverted_index)