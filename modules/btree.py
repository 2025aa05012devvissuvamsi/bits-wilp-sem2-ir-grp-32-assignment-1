"""
B-Tree implementation for the
Information Retrieval system.

This module implements:
    1. B-Tree node structure
    2. B-Tree insertion
    3. B-Tree search
    4. Vocabulary dictionary creation

The implementation is intentionally simplified
for educational demonstration and experimental
comparison against Binary Search Trees.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional


# ============================================================================
# B-TREE NODE DEFINITION
# ============================================================================

@dataclass
class BTreeNode:
    """
    Node structure for B-Tree.

    Attributes:
        leaf:
            Indicates whether node is leaf node.

        keys:
            Sorted key collection stored in node.

        children:
            Child node references.
    """

    leaf: bool = True

    keys: List[str] = field(
        default_factory=list
    )

    children: List["BTreeNode"] = field(
        default_factory=list
    )


# ============================================================================
# B-TREE IMPLEMENTATION
# ============================================================================

class BTree:
    """
    Simplified B-Tree implementation for
    vocabulary dictionary storage.

    Supported operations:
        1. Insert term
        2. Search term

    Default minimum degree:
        t = 2
    """

    def __init__(
        self,
        minimum_degree: int = 2,
    ) -> None:
        """
        Initialize B-Tree structure.

        Args:
            minimum_degree:
                Minimum degree of the B-Tree.
        """
        self.minimum_degree: int = minimum_degree

        self.root: BTreeNode = BTreeNode()

    # ------------------------------------------------------------------------
    # B-TREE SEARCH
    # ------------------------------------------------------------------------

    def search(
        self,
        value: str,
        node: Optional[BTreeNode] = None,
    ) -> bool:
        """
        Search for a term in the B-Tree.

        Args:
            value:
                Vocabulary term to search.

            node:
                Current traversal node.

        Returns:
            bool:
                True if term exists,
                otherwise False.
        """
        if node is None:
            node = self.root

        index: int = 0

        while (
            index < len(node.keys)
            and value > node.keys[index]
        ):
            index += 1

        if (
            index < len(node.keys)
            and value == node.keys[index]
        ):
            return True

        if node.leaf:
            return False

        return self.search(
            value=value,
            node=node.children[index],
        )

    # ------------------------------------------------------------------------
    # B-TREE INSERTION
    # ------------------------------------------------------------------------

    def insert(
        self,
        value: str,
    ) -> None:
        """
        Insert a vocabulary term into the B-Tree.

        Args:
            value:
                Vocabulary term to insert.

        Returns:
            None
        """
        root_node: BTreeNode = self.root

        maximum_keys: int = (
            2 * self.minimum_degree
        ) - 1

        if len(root_node.keys) == maximum_keys:

            new_root: BTreeNode = BTreeNode(
                leaf=False
            )

            new_root.children.append(
                root_node
            )

            self._split_child(
                parent_node=new_root,
                child_index=0,
            )

            self.root = new_root

        self._insert_non_full(
            node=self.root,
            value=value,
        )

    def _insert_non_full(
        self,
        node: BTreeNode,
        value: str,
    ) -> None:
        """
        Insert a value into a non-full node.

        Args:
            node:
                Target node.

            value:
                Vocabulary term to insert.

        Returns:
            None
        """
        index: int = len(node.keys) - 1

        if node.leaf:

            node.keys.append("")

            while (
                index >= 0
                and value < node.keys[index]
            ):
                node.keys[index + 1] = (
                    node.keys[index]
                )

                index -= 1

            node.keys[index + 1] = value

            return

        while (
            index >= 0
            and value < node.keys[index]
        ):
            index -= 1

        index += 1

        maximum_keys: int = (
            2 * self.minimum_degree
        ) - 1

        if (
            len(node.children[index].keys)
            == maximum_keys
        ):
            self._split_child(
                parent_node=node,
                child_index=index,
            )

            if value > node.keys[index]:
                index += 1

        self._insert_non_full(
            node=node.children[index],
            value=value,
        )

    # ------------------------------------------------------------------------
    # CHILD SPLITTING
    # ------------------------------------------------------------------------

    def _split_child(
        self,
        parent_node: BTreeNode,
        child_index: int,
    ) -> None:
        """
        Split a full child node.

        Args:
            parent_node:
                Parent node reference.

            child_index:
                Index of child to split.

        Returns:
            None
        """
        minimum_degree: int = self.minimum_degree

        full_child: BTreeNode = (
            parent_node.children[child_index]
        )

        new_child: BTreeNode = BTreeNode(
            leaf=full_child.leaf
        )

        middle_key: str = (
            full_child.keys[minimum_degree - 1]
        )

        new_child.keys = (
            full_child.keys[minimum_degree:]
        )

        full_child.keys = (
            full_child.keys[:minimum_degree - 1]
        )

        if not full_child.leaf:

            new_child.children = (
                full_child.children[minimum_degree:]
            )

            full_child.children = (
                full_child.children[:minimum_degree]
            )

        parent_node.children.insert(
            child_index + 1,
            new_child,
        )

        parent_node.keys.insert(
            child_index,
            middle_key,
        )


# ============================================================================
# B-TREE CONSTRUCTION
# ============================================================================

def build_btree_from_vocabulary(
    vocabulary_terms: List[str],
) -> BTree:
    """
    Build B-Tree from vocabulary terms.

    Args:
        vocabulary_terms:
            Vocabulary term collection.

    Returns:
        BTree:
            Constructed B-Tree instance.
    """
    btree: BTree = BTree()

    for term in vocabulary_terms:

        btree.insert(
            value=term
        )

    return btree