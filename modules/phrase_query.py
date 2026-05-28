"""
Phrase query processing module for the
Information Retrieval system.

This module implements:
    1. Biword index creation
    2. Positional index creation
    3. Phrase query search
    4. Phrase retrieval comparison
    5. False positive analysis

The implementations are intentionally designed
for educational visualization and experimental
comparison within the Streamlit application.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from collections import defaultdict
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Set

from pandas import DataFrame

from modules.preprocessing import preprocess_text


# ============================================================================
# BIWORD INDEX CREATION
# ============================================================================

def create_biword_index(
    documents_df: DataFrame,
) -> Dict[str, List[int]]:
    """
    Create a biword index from the document collection.

    A biword index stores consecutive token pairs
    mapped to document identifiers.

    Example:
        \"machine learning techniques\"

    Generated biwords:
        - machine learning
        - learning techniques

    Index structure:
        biword -> [document_ids]

    Args:
        documents_df:
            Input document collection dataframe.

    Returns:
        Dict[str, List[int]]:
            Generated biword index.
    """
    biword_index: DefaultDict[str, Set[int]] = defaultdict(set)

    for _, row in documents_df.iterrows():

        document_id: int = int(row["doc_id"])

        document_text: str = str(row["text"])

        tokens: List[str] = preprocess_text(
            text=document_text
        )

        for index in range(len(tokens) - 1):

            biword: str = (
                f"{tokens[index]} "
                f"{tokens[index + 1]}"
            )

            biword_index[biword].add(document_id)

    return {
        biword: sorted(list(document_ids))
        for biword, document_ids
        in biword_index.items()
    }


# ============================================================================
# POSITIONAL INDEX CREATION
# ============================================================================

def create_positional_index(
    documents_df: DataFrame,
) -> Dict[str, Dict[int, List[int]]]:
    """
    Create a positional index from the document collection.

    Positional indexes store:
        term -> document_id -> positions

    Example:
        {
            \"machine\": {
                1: [0],
                6: [0],
            }
        }

    Args:
        documents_df:
            Input document collection dataframe.

    Returns:
        Dict[str, Dict[int, List[int]]]:
            Generated positional index.
    """
    positional_index: DefaultDict[
        str,
        DefaultDict[int, List[int]]
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for _, row in documents_df.iterrows():

        document_id: int = int(row["doc_id"])

        document_text: str = str(row["text"])

        tokens: List[str] = preprocess_text(
            text=document_text
        )

        for position, token in enumerate(tokens):

            positional_index[token][document_id].append(
                position
            )

    return {
        token: dict(document_positions)
        for token, document_positions
        in positional_index.items()
    }


# ============================================================================
# BIWORD PHRASE QUERY SEARCH
# ============================================================================

def search_biword_phrase(
    phrase_query: str,
    biword_index: Dict[str, List[int]],
) -> List[int]:
    """
    Execute phrase query retrieval using biword index.

    Retrieval logic:
        - Convert phrase query into biwords
        - Intersect matching document sets

    Args:
        phrase_query:
            Input phrase query.

        biword_index:
            Generated biword index.

    Returns:
        List[int]:
            Retrieved document identifiers.
    """
    query_tokens: List[str] = preprocess_text(
        text=phrase_query
    )

    if len(query_tokens) < 2:
        return []

    query_biwords: List[str] = []

    for index in range(len(query_tokens) - 1):

        biword: str = (
            f"{query_tokens[index]} "
            f"{query_tokens[index + 1]}"
        )

        query_biwords.append(biword)

    matching_documents: List[Set[int]] = []

    for biword in query_biwords:

        document_ids: Set[int] = set(
            biword_index.get(biword, [])
        )

        matching_documents.append(document_ids)

    if not matching_documents:
        return []

    return sorted(
        list(
            set.intersection(*matching_documents)
        )
    )


# ============================================================================
# POSITIONAL PHRASE QUERY SEARCH
# ============================================================================

def search_positional_phrase(
    phrase_query: str,
    positional_index: Dict[str, Dict[int, List[int]]],
) -> List[int]:
    """
    Execute phrase query retrieval using positional index.

    Positional retrieval validates exact sequential
    adjacency between query terms.

    Args:
        phrase_query:
            Input phrase query.

        positional_index:
            Generated positional index.

    Returns:
        List[int]:
            Retrieved document identifiers.
    """
    query_tokens: List[str] = preprocess_text(
        text=phrase_query
    )

    if len(query_tokens) < 2:
        return []

    first_term: str = query_tokens[0]

    candidate_documents: Set[int] = set(
        positional_index.get(first_term, {}).keys()
    )

    matching_documents: List[int] = []

    for document_id in candidate_documents:

        first_positions: List[int] = (
            positional_index[first_term][document_id]
        )

        for start_position in first_positions:

            phrase_match: bool = True

            for offset in range(1, len(query_tokens)):

                current_term: str = query_tokens[offset]

                current_positions: List[int] = (
                    positional_index.get(
                        current_term,
                        {}
                    ).get(document_id, [])
                )

                expected_position: int = (
                    start_position + offset
                )

                if expected_position not in current_positions:

                    phrase_match = False
                    break

            if phrase_match:

                matching_documents.append(document_id)
                break

    return sorted(matching_documents)


# ============================================================================
# FALSE POSITIVE ANALYSIS
# ============================================================================

def generate_phrase_query_inference(
    biword_results: List[int] = None,
    positional_results: List[int] = None,
    phrase_query: str = "",
) -> str:
    """
    Generate inference explaining the difference
    between biword and positional indexes, with
    context from actual query results.

    Args:
        biword_results:
            Document IDs retrieved via biword index.

        positional_results:
            Document IDs retrieved via positional index.

        phrase_query:
            The original phrase query string.

    Returns:
        str:
            Generated inference statement.
    """
    biword_results = biword_results or []
    positional_results = positional_results or []

    biword_count = len(biword_results)
    positional_count = len(positional_results)

    false_positives = set(biword_results) - set(positional_results)
    false_positive_count = len(false_positives)

    lines = []

    lines.append(
        f"Biword Index returned {biword_count} document(s); "
        f"Positional Index returned {positional_count} document(s)."
    )

    if false_positive_count > 0:
        lines.append(
            f"False positives from Biword Index: {false_positive_count} "
            f"document(s) (Doc IDs: {sorted(false_positives)}). "
            "These documents contain the individual adjacent token pairs "
            "but the complete phrase may not appear verbatim — a known "
            "limitation of biword indexing for longer queries."
        )
    else:
        lines.append(
            "No false positives detected for this query, meaning both "
            "indexes agree on all matched documents. For longer or more "
            "complex phrase queries, biword indexes are more likely to "
            "produce false positives."
        )

    lines.append(
        "Why Positional Index is more accurate: It validates exact "
        "token adjacency by storing per-document term positions. The "
        "retrieval algorithm confirms that each consecutive query term "
        "appears exactly one position after the previous, making it "
        "immune to false positives that biword indexes can produce."
    )

    lines.append(
        "Trade-off: Biword indexes are faster to build and query for "
        "short 2-gram phrases but scale poorly with phrase length. "
        "Positional indexes require more storage but provide exact "
        "phrase semantics regardless of phrase length."
    )

    return " | ".join(lines)
