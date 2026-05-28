"""
Binary Search Tree implementation for the
Information Retrieval system.

This module implements:
    1. BST node structure
    2. BST insertion
    3. BST search
    4. Vocabulary dictionary creation
    5. Experimental search utilities

The implementation is intentionally designed
for educational demonstration and performance
comparison against B-Tree structures.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Set

from pandas import DataFrame

from modules.preprocessing import preprocess_text


# ============================================================================
# BST NODE DEFINITION
# ============================================================================

@dataclass
class BSTNode:
    """
    Node structure for Binary Search Tree.

    Attributes:
        value:
            Vocabulary term stored in the node.

        left:
            Reference to left child node.

        right:
            Reference to right child node.
    """

    value: str

    left: Optional["BSTNode"] = None

    right: Optional["BSTNode"] = None


# ============================================================================
# BINARY SEARCH TREE IMPLEMENTATION
# ============================================================================

class BinarySearchTree:
    """
    Binary Search Tree implementation for
    dictionary term storage and lookup.

    Supported operations:
        1. Insert term
        2. Search term
        3. Build vocabulary dictionary
    """

    def __init__(self) -> None:
        """
        Initialize empty Binary Search Tree.
        """
        self.root: Optional[BSTNode] = None

    # ------------------------------------------------------------------------
    # BST INSERTION
    # ------------------------------------------------------------------------

    def insert(
        self,
        value: str,
    ) -> None:
        """
        Insert a vocabulary term into the BST.

        Args:
            value:
                Vocabulary term to insert.

        Returns:
            None
        """
        if self.root is None:

            self.root = BSTNode(
                value=value
            )

            return

        self._insert_recursive(
            current_node=self.root,
            value=value,
        )

    def _insert_recursive(
        self,
        current_node: BSTNode,
        value: str,
    ) -> None:
        """
        Recursively insert a value into the BST.

        Args:
            current_node:
                Current traversal node.

            value:
                Vocabulary term to insert.

        Returns:
            None
        """
        if value < current_node.value:

            if current_node.left is None:

                current_node.left = BSTNode(
                    value=value
                )

                return

            self._insert_recursive(
                current_node=current_node.left,
                value=value,
            )

        elif value > current_node.value:

            if current_node.right is None:

                current_node.right = BSTNode(
                    value=value
                )

                return

            self._insert_recursive(
                current_node=current_node.right,
                value=value,
            )

    # ------------------------------------------------------------------------
    # BST SEARCH
    # ------------------------------------------------------------------------

    def search(
        self,
        value: str,
    ) -> bool:
        """
        Search for a vocabulary term in the BST.

        Args:
            value:
                Vocabulary term to search.

        Returns:
            bool:
                True if term exists,
                otherwise False.
        """
        return self._search_recursive(
            current_node=self.root,
            value=value,
        )

    def _search_recursive(
        self,
        current_node: Optional[BSTNode],
        value: str,
    ) -> bool:
        """
        Recursively search for a value in BST.

        Args:
            current_node:
                Current traversal node.

            value:
                Vocabulary term to search.

        Returns:
            bool:
                Search status.
        """
        if current_node is None:
            return False

        if current_node.value == value:
            return True

        if value < current_node.value:

            return self._search_recursive(
                current_node=current_node.left,
                value=value,
            )

        return self._search_recursive(
            current_node=current_node.right,
            value=value,
        )


# ============================================================================
# VOCABULARY EXTRACTION
# ============================================================================

def extract_vocabulary(
    documents_df: DataFrame,
) -> List[str]:
    """
    Extract unique vocabulary terms from
    the document collection.

    Processing workflow:
        1. Preprocess each document
        2. Extract tokens
        3. Build unique vocabulary set
        4. Return sorted vocabulary

    Args:
        documents_df:
            Input document collection dataframe.

    Returns:
        List[str]:
            Sorted vocabulary terms.
    """
    vocabulary_terms: Set[str] = set()

    for _, row in documents_df.iterrows():

        document_text: str = str(row["text"])

        processed_tokens: List[str] = preprocess_text(
            text=document_text
        )

        vocabulary_terms.update(
            processed_tokens
        )

    return sorted(
        list(vocabulary_terms)
    )


# ============================================================================
# BST CONSTRUCTION
# ============================================================================

def build_bst_from_vocabulary(
    vocabulary_terms: List[str],
) -> BinarySearchTree:
    """
    Build Binary Search Tree from vocabulary terms.

    Args:
        vocabulary_terms:
            Vocabulary term collection.

    Returns:
        BinarySearchTree:
            Constructed BST instance.
    """
    bst: BinarySearchTree = BinarySearchTree()

    for term in vocabulary_terms:

        bst.insert(
            value=term
        )

    return bst