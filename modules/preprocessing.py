"""
Text preprocessing module for the Information Retrieval system.

This module contains reusable preprocessing utilities required
for document normalization and token preparation before indexing
and retrieval operations.

Implemented preprocessing operations:
    1. Tokenization
    2. Lowercasing
    3. Punctuation removal
    4. Stop word removal
    5. Hyphen normalization
    6. Stemming
    7. Lemmatization
"""

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK data at import time
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

STOP_WORDS: set[str] = set(stopwords.words("english"))

STEMMER = PorterStemmer()

LEMMATIZER = WordNetLemmatizer()


def tokenize_text(text: str) -> List[str]:
    """
    Convert raw text into individual tokens.

    Tokenization is the foundational preprocessing step
    where a text sequence is segmented into smaller units
    called tokens.

    Args:
        text:
            Raw input document text.

    Returns:
        List[str]:
            List containing extracted tokens.
    """
    return word_tokenize(text=text)


def convert_to_lowercase(tokens: List[str]) -> List[str]:
    """
    Convert all tokens into lowercase representation.

    Lowercasing helps normalize vocabulary by ensuring
    words with different cases are treated as identical
    terms during indexing and retrieval.

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Lowercased token list.
    """
    return [token.lower() for token in tokens]


def remove_punctuation(tokens: List[str]) -> List[str]:
    """
    Remove punctuation tokens from the token sequence.

    Punctuation marks (periods, commas, quotes, etc.) carry
    no semantic value for retrieval and are discarded to keep
    the vocabulary clean.

    Strategy:
        - Tokens that consist entirely of non-alphanumeric
          characters are dropped.
        - Trailing/leading punctuation is stripped from
          alphanumeric tokens (e.g. \"don't\" -> \"don't\"
          is kept, but \".\" is dropped).

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Cleaned token list with punctuation removed.
    """
    cleaned_tokens: List[str] = []

    for token in tokens:

        # Strip surrounding punctuation characters
        stripped: str = token.strip(
            ".,!?;:\"'()[]{}<>|\\/@#$%^&*+=~`"
        )

        # Keep token only if it has at least one alphanumeric character
        if re.search(r"[A-Za-z0-9]", stripped):
            cleaned_tokens.append(stripped)

    return cleaned_tokens


def handle_hyphenated_words(tokens: List[str]) -> List[str]:
    """
    Normalize hyphenated words by replacing hyphens
    with whitespace-separated terms.

    Example:
        \"state-of-the-art\"
        becomes
        [\"state\", \"of\", \"the\", \"art\"]

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Normalized token list.
    """
    normalized_tokens: List[str] = []

    for token in tokens:
        split_tokens: List[str] = token.replace(
            "-",
            " "
        ).split()

        normalized_tokens.extend(split_tokens)

    return normalized_tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common stopwords from the token sequence.

    Stopwords are high-frequency terms that generally
    contribute minimal semantic value during retrieval.

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Filtered token list without stopwords.
    """
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def apply_stemming(tokens: List[str]) -> List[str]:
    """
    Apply Porter stemming algorithm to tokens.

    Stemming reduces words to their root forms
    using heuristic suffix stripping.

    Example:
        \"running\" -> \"run\"

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Stemmed token list.
    """
    return [
        STEMMER.stem(token)
        for token in tokens
    ]


def apply_lemmatization(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to normalize tokens into
    linguistically valid base forms.

    Unlike stemming, lemmatization considers
    vocabulary and morphological analysis.

    Example:
        \"better\" -> \"good\"

    Args:
        tokens:
            Input token list.

    Returns:
        List[str]:
            Lemmatized token list.
    """
    return [
        LEMMATIZER.lemmatize(token)
        for token in tokens
    ]


def preprocess_text(
    text: str,
    apply_stemmer: bool = False,
    apply_lemmatizer: bool = False,
) -> List[str]:
    """
    Execute the complete preprocessing pipeline.

    Processing stages:
        1. Tokenization
        2. Lowercasing
        3. Punctuation removal
        4. Hyphen normalization
        5. Stopword removal
        6. Optional stemming
        7. Optional lemmatization

    Args:
        text:
            Raw document text.

        apply_stemmer:
            Whether stemming should be applied.

        apply_lemmatizer:
            Whether lemmatization should be applied.

    Returns:
        List[str]:
            Fully processed token sequence.
    """
    tokens: List[str] = tokenize_text(text=text)

    tokens = convert_to_lowercase(tokens=tokens)

    tokens = remove_punctuation(tokens=tokens)

    tokens = handle_hyphenated_words(tokens=tokens)

    tokens = remove_stopwords(tokens=tokens)

    if apply_stemmer:
        tokens = apply_stemming(tokens=tokens)

    if apply_lemmatizer:
        tokens = apply_lemmatization(tokens=tokens)

    return tokens
