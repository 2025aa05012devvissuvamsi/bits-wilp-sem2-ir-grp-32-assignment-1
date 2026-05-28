"""
Tolerant retrieval module for the
Information Retrieval system.

This module implements:
    1. K-Gram indexing
    2. Wildcard query matching
    3. Edit distance computation
    4. Spelling correction
    5. Phonetic correction (Soundex)
    6. Approximate term retrieval

The implementation is intentionally designed
for educational visualization and experimental
demonstration within the Streamlit application.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from collections import defaultdict
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from pandas import DataFrame

from modules.bst import extract_vocabulary


# ============================================================================
# K-GRAM INDEX CREATION
# ============================================================================

def generate_kgrams(
    term: str,
    k: int = 3,
) -> List[str]:
    """
    Generate k-grams for a vocabulary term.

    Boundary markers are added to improve
    prefix and suffix matching.

    Example:
        retrieval -> ['$re', 'ret', 'etr', ...]

    Args:
        term:
            Input vocabulary term.

        k:
            Gram size.

    Returns:
        List[str]:
            Generated k-gram list.
    """
    padded_term: str = f"${term}$"

    kgrams: List[str] = []

    for index in range(
        len(padded_term) - k + 1
    ):
        kgram: str = padded_term[
            index:index + k
        ]

        kgrams.append(kgram)

    return kgrams


def create_kgram_index(
    vocabulary_terms: List[str],
    k: int = 3,
) -> Dict[str, List[str]]:
    """
    Create K-Gram index from vocabulary terms.

    Index structure:
        kgram -> matching terms

    Args:
        vocabulary_terms:
            Vocabulary collection.

        k:
            Gram size.

    Returns:
        Dict[str, List[str]]:
            Generated K-Gram index.
    """
    kgram_index: DefaultDict[
        str,
        Set[str]
    ] = defaultdict(set)

    for term in vocabulary_terms:

        term_kgrams: List[str] = generate_kgrams(
            term=term,
            k=k,
        )

        for kgram in term_kgrams:

            kgram_index[kgram].add(term)

    return {
        kgram: sorted(list(terms))
        for kgram, terms
        in kgram_index.items()
    }


# ============================================================================
# EDIT DISTANCE COMPUTATION
# ============================================================================

def compute_edit_distance(
    source_term: str,
    target_term: str,
) -> int:
    """
    Compute Levenshtein edit distance between
    two terms.

    Supported operations:
        1. Insertion
        2. Deletion
        3. Substitution

    Args:
        source_term:
            Source vocabulary term.

        target_term:
            Target vocabulary term.

    Returns:
        int:
            Computed edit distance.
    """
    source_length: int = len(source_term)

    target_length: int = len(target_term)

    distance_matrix: List[List[int]] = [
        [0] * (target_length + 1)
        for _ in range(source_length + 1)
    ]

    for row in range(source_length + 1):
        distance_matrix[row][0] = row

    for column in range(target_length + 1):
        distance_matrix[0][column] = column

    for row in range(1, source_length + 1):

        for column in range(1, target_length + 1):

            substitution_cost: int = (
                0
                if source_term[row - 1]
                == target_term[column - 1]
                else 1
            )

            distance_matrix[row][column] = min(
                distance_matrix[row - 1][column] + 1,
                distance_matrix[row][column - 1] + 1,
                distance_matrix[row - 1][column - 1]
                + substitution_cost,
            )

    return distance_matrix[
        source_length
    ][target_length]


# ============================================================================
# SPELLING CORRECTION
# ============================================================================

def suggest_spelling_corrections(
    query_term: str,
    vocabulary_terms: List[str],
    max_suggestions: int = 5,
) -> List[Tuple[str, int]]:
    """
    Generate spelling correction suggestions
    using edit distance similarity.

    Args:
        query_term:
            User-entered query term.

        vocabulary_terms:
            Vocabulary collection.

        max_suggestions:
            Maximum suggestions to return.

    Returns:
        List[Tuple[str, int]]:
            Suggested terms with edit distance.
    """
    candidate_terms: List[
        Tuple[str, int]
    ] = []

    for vocabulary_term in vocabulary_terms:

        distance: int = compute_edit_distance(
            source_term=query_term,
            target_term=vocabulary_term,
        )

        candidate_terms.append(
            (
                vocabulary_term,
                distance,
            )
        )

    candidate_terms.sort(
        key=lambda item: item[1]
    )

    return candidate_terms[:max_suggestions]


# ============================================================================
# PHONETIC CORRECTION (SOUNDEX)
# ============================================================================

def compute_soundex(term: str) -> str:
    """
    Compute the Soundex phonetic code for a term.

    Soundex encodes a word to a 4-character code
    representing its English pronunciation, allowing
    retrieval of phonetically similar terms even when
    the spelling differs significantly.

    Algorithm:
        1. Retain the first letter.
        2. Replace consonants with digits per the
           Soundex mapping table.
        3. Remove adjacent duplicates and zeros.
        4. Pad or truncate to length 4.

    Example:
        \"Robert\" -> \"R163\"
        \"Rupert\" -> \"R163\"

    Args:
        term:
            Input vocabulary term.

    Returns:
        str:
            4-character Soundex code.
    """
    if not term:
        return "0000"

    term = term.upper()

    soundex_map: Dict[str, str] = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2",
        "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    first_letter: str = term[0]

    encoded: str = first_letter

    previous_code: str = soundex_map.get(
        first_letter, "0"
    )

    for char in term[1:]:

        code: str = soundex_map.get(char, "0")

        if code != "0" and code != previous_code:
            encoded += code

        previous_code = code

        if len(encoded) == 4:
            break

    return encoded.ljust(4, "0")


def suggest_phonetic_corrections(
    query_term: str,
    vocabulary_terms: List[str],
    max_suggestions: int = 5,
) -> List[str]:
    """
    Suggest vocabulary terms that are phonetically
    similar to the query term using Soundex.

    Args:
        query_term:
            User-entered query term.

        vocabulary_terms:
            Vocabulary collection.

        max_suggestions:
            Maximum suggestions to return.

    Returns:
        List[str]:
            Phonetically matching vocabulary terms.
    """
    query_soundex: str = compute_soundex(query_term)

    phonetic_matches: List[str] = [
        term
        for term in vocabulary_terms
        if compute_soundex(term) == query_soundex
        and term != query_term
    ]

    return sorted(phonetic_matches)[:max_suggestions]


# ============================================================================
# WILDCARD QUERY MATCHING
# ============================================================================

def perform_wildcard_search(
    wildcard_query: str,
    vocabulary_terms: List[str],
) -> List[str]:
    """
    Perform wildcard matching against vocabulary.

    Supported wildcard:
        *

    Example:
        retr*

    Args:
        wildcard_query:
            Wildcard query pattern.

        vocabulary_terms:
            Vocabulary collection.

    Returns:
        List[str]:
            Matching vocabulary terms.
    """
    wildcard_pattern: str = wildcard_query.replace(
        "*",
        ""
    )

    matching_terms: List[str] = []

    for vocabulary_term in vocabulary_terms:

        if vocabulary_term.startswith(
            wildcard_pattern
        ):
            matching_terms.append(
                vocabulary_term
            )

    return sorted(matching_terms)


# ============================================================================
# TOLERANT RETRIEVAL EXPERIMENT
# ============================================================================

def execute_tolerant_retrieval_experiment(
    documents_df: DataFrame,
    query_term: str,
) -> Dict[str, object]:
    """
    Execute tolerant retrieval experiment.

    Workflow:
        1. Extract vocabulary
        2. Build K-Gram index
        3. Generate spelling suggestions
        4. Execute wildcard matching
        5. Phonetic correction via Soundex

    Args:
        documents_df:
            Input document collection dataframe.

        query_term:
            User-entered query term.

    Returns:
        Dict[str, object]:
            Experimental retrieval outputs.
    """
    vocabulary_terms: List[str] = extract_vocabulary(
        documents_df=documents_df
    )

    kgram_index: Dict[str, List[str]] = (
        create_kgram_index(
            vocabulary_terms=vocabulary_terms
        )
    )

    spelling_suggestions: List[
        Tuple[str, int]
    ] = suggest_spelling_corrections(
        query_term=query_term,
        vocabulary_terms=vocabulary_terms,
    )

    wildcard_matches: List[str] = (
        perform_wildcard_search(
            wildcard_query=query_term,
            vocabulary_terms=vocabulary_terms,
        )
    )

    query_kgrams: List[str] = generate_kgrams(
        term=query_term
    )

    phonetic_matches: List[str] = (
        suggest_phonetic_corrections(
            query_term=query_term,
            vocabulary_terms=vocabulary_terms,
        )
    )

    return {
        "query_term": query_term,
        "query_kgrams": query_kgrams,
        "kgram_index": kgram_index,
        "spelling_suggestions": spelling_suggestions,
        "wildcard_matches": wildcard_matches,
        "phonetic_matches": phonetic_matches,
        "query_soundex": compute_soundex(query_term),
    }
