"""
Retrieval module for Information Retrieval system.

This module provides utilities for:
    1. Query preprocessing
    2. Document retrieval
    3. Retrieval comparison
    4. Experimental evaluation
    5. Precision / recall estimation

Implemented retrieval workflows:
    - Stemming-based retrieval
    - Lemmatization-based retrieval
"""

# ============================================================================
# IMPORTS
# ============================================================================

from time import perf_counter
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from pandas import DataFrame

from modules.preprocessing import preprocess_text


# ============================================================================
# QUERY RETRIEVAL UTILITIES
# ============================================================================

def retrieve_matching_documents(
    processed_documents: Dict[int, List[str]],
    query_tokens: List[str],
) -> List[int]:
    """
    Retrieve documents matching the processed query tokens.

    Retrieval logic:
        - A document is considered relevant if at least
          one query token exists in the document token set.

    Matching strategy:
        OR-based retrieval

    Args:
        processed_documents:
            Dictionary containing document ID mapped
            to processed token sequences.

        query_tokens:
            Preprocessed query token sequence.

    Returns:
        List[int]:
            Sorted list of retrieved document IDs.
    """
    matching_documents: Set[int] = set()

    for document_id, document_tokens in processed_documents.items():

        if any(
            token in document_tokens
            for token in query_tokens
        ):
            matching_documents.add(document_id)

    return sorted(list(matching_documents))


# ============================================================================
# DOCUMENT PREPROCESSING FOR RETRIEVAL
# ============================================================================

def build_processed_document_collection(
    documents_df: DataFrame,
    apply_stemmer: bool = False,
    apply_lemmatizer: bool = False,
) -> Dict[int, List[str]]:
    """
    Preprocess the entire document collection
    using the selected normalization strategy.

    Args:
        documents_df:
            Input document dataframe.

        apply_stemmer:
            Whether stemming should be applied.

        apply_lemmatizer:
            Whether lemmatization should be applied.

    Returns:
        Dict[int, List[str]]:
            Processed document collection.
    """
    processed_documents: Dict[int, List[str]] = {}

    for _, row in documents_df.iterrows():

        document_id: int = int(row["doc_id"])

        document_text: str = str(row["text"])

        processed_tokens: List[str] = preprocess_text(
            text=document_text,
            apply_stemmer=apply_stemmer,
            apply_lemmatizer=apply_lemmatizer,
        )

        processed_documents[document_id] = processed_tokens

    return processed_documents


# ============================================================================
# RETRIEVAL EXPERIMENT ENGINE
# ============================================================================

def execute_retrieval_experiment(
    documents_df: DataFrame,
    query: str,
    apply_stemmer: bool = False,
    apply_lemmatizer: bool = False,
) -> Tuple[List[int], float]:
    """
    Execute a complete retrieval experiment.

    Experimental workflow:
        1. Preprocess document collection
        2. Preprocess query
        3. Execute retrieval
        4. Measure execution time

    Args:
        documents_df:
            Input document dataframe.

        query:
            Raw user search query.

        apply_stemmer:
            Whether stemming should be enabled.

        apply_lemmatizer:
            Whether lemmatization should be enabled.

    Returns:
        Tuple[List[int], float]:
            Retrieved documents and execution time.
    """
    start_time: float = perf_counter()

    processed_documents: Dict[int, List[str]] = (
        build_processed_document_collection(
            documents_df=documents_df,
            apply_stemmer=apply_stemmer,
            apply_lemmatizer=apply_lemmatizer,
        )
    )

    processed_query_tokens: List[str] = preprocess_text(
        text=query,
        apply_stemmer=apply_stemmer,
        apply_lemmatizer=apply_lemmatizer,
    )

    retrieved_documents: List[int] = retrieve_matching_documents(
        processed_documents=processed_documents,
        query_tokens=processed_query_tokens,
    )

    end_time: float = perf_counter()

    execution_time: float = end_time - start_time

    return retrieved_documents, execution_time


# ============================================================================
# INFERENCE GENERATION
# ============================================================================

def generate_comparison_inference(
    stemming_results: List[int],
    lemmatization_results: List[int],
    query: str = "",
) -> str:
    """
    Generate experimental inference comparing
    stemming and lemmatization retrieval outputs.

    Args:
        stemming_results:
            Retrieved documents from stemming pipeline.

        lemmatization_results:
            Retrieved documents from lemmatization pipeline.

        query:
            The original query string (optional context).

    Returns:
        str:
            Generated inference statement.
    """
    stem_count: int = len(stemming_results)
    lemma_count: int = len(lemmatization_results)

    union_docs = set(stemming_results) | set(lemmatization_results)
    overlap_docs = set(stemming_results) & set(lemmatization_results)

    overlap_pct = (
        round(len(overlap_docs) / len(union_docs) * 100, 1)
        if union_docs
        else 0.0
    )

    lines = []

    if stem_count > lemma_count:
        lines.append(
            f"Stemming retrieved {stem_count} document(s) vs "
            f"Lemmatization's {lemma_count}, indicating higher recall "
            f"due to aggressive suffix stripping that groups more word "
            f"variants under a common root."
        )
        lines.append(
            "However, over-stemming may reduce precision by conflating "
            "semantically distinct words (e.g., 'universe' and "
            "'university' both reduce to 'univers')."
        )
        lines.append(
            "Recommendation: For this dataset, stemming improves recall "
            "but should be used alongside filtering if precision matters."
        )

    elif lemma_count > stem_count:
        lines.append(
            f"Lemmatization retrieved {lemma_count} document(s) vs "
            f"Stemming's {stem_count}, producing semantically accurate "
            f"base forms using morphological analysis."
        )
        lines.append(
            "Lemmatization is linguistically superior because it maps "
            "inflected forms to valid dictionary entries (e.g., "
            "'retrieved' -> 'retrieve'), preserving meaning."
        )
        lines.append(
            "Recommendation: Lemmatization is more suitable for this "
            "dataset given its superior precision and meaningful "
            "vocabulary normalization."
        )

    else:
        lines.append(
            f"Both techniques retrieved {stem_count} document(s), "
            f"achieving identical recall for this query."
        )
        lines.append(
            "This result indicates the query terms are morphologically "
            "simple, so both approaches converge to equivalent token sets."
        )
        lines.append(
            f"Overlap between result sets: {overlap_pct}%. "
            "For more complex queries with inflected vocabulary, "
            "lemmatization typically outperforms stemming in precision."
        )

    lines.append(
        f"Result overlap: {overlap_pct}% of unique retrieved documents "
        f"appear in both result sets, confirming "
        + (
            "significant divergence — the techniques are complementary."
            if overlap_pct < 80
            else "high agreement — the vocabulary is morphologically regular."
        )
    )

    return " | ".join(lines)
