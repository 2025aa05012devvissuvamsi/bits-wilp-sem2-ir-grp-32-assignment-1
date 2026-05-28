"""
Application-wide constants for the
Information Retrieval system.

This module stores:
    1. Application metadata
    2. Group information
    3. Supported file formats
    4. Global configuration constants
"""

# ============================================================================
# APPLICATION METADATA
# ============================================================================

APP_TITLE: str = (
    "Information Retrieval System"
)

APP_DESCRIPTION: str = (
    "Interactive Information Retrieval "
    "System using Streamlit"
)

# ============================================================================
# GROUP INFORMATION
# ============================================================================

GROUP_NUMBER: str = "32"

GROUP_MEMBERS = [
    {
        "registration": "2025aa05005",
        "name": "PRAVEEN YADAV",
        "contribution": "100%",
    },
    {
        "registration": "2025aa05011",
        "name": "RONGALI BHAVYA TEJA",
        "contribution": "100%",
    },
    {
        "registration": "2025aa05012",
        "name": "P VAMSI KRISHNA",
        "contribution": "100%",
    },
]

COURSE_NAME: str = (
    "Information Retrieval"
)

ASSIGNMENT_TITLE: str = "Assignment - 1"

# ============================================================================
# SUPPORTED FILE FORMATS
# ============================================================================

SUPPORTED_FILE_TYPES = [
    "csv",
    "pdf",
]

# ============================================================================
# DEFAULT APPLICATION SETTINGS
# ============================================================================

DEFAULT_QUERY: str = (
    "machine learning"
)

DEFAULT_PHRASE_QUERY: str = (
    "machine learning"
)

DEFAULT_DICTIONARY_TERM: str = (
    "retrieval"
)

DEFAULT_TOLERANT_QUERY: str = (
    "retrival"
)
