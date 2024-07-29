"""The neighbor lookup class facilitates looking up the neighboring nodes along
the adjacent entities, i.e., faces or tetrahedra, of a given node.
"""

from typing import Any

import numpy as np


class NeighborLookup:
    """Neighbor lookup.

    The neighbors are stored in an array, such that each node index is
    associated with an array of length equal to the number of adjacent entities.
    This array contains 2-tuples in the case of triangles or 3-tuples in the
    case of tetrahedra containing the neighboring node tags.
    """

    def __init__(self, node_tags: np.ndarray) -> None:
        self.neighbors = [[] for _ in range(np.max(node_tags))]
        self._init_neighbors(node_tags)

    def get_num_adjacent_entities(self, tag: int) -> int:
        """Returns the number of adjacent entities.

        Args:
            tag: Node tag.
        """
        node_index = self._get_index_from_tag(tag)
        return len(self.neighbors[node_index])

    def get_neighbors(self, tag: int) -> list[np.ndarray]:
        """Returns the neighbors of the given node.

        Args:
            tag: Node tag.

        Returns:
            A list consisting of arrays containing the neighboring node tags.
        """
        node_index = self._get_index_from_tag(tag)
        return self.neighbors[node_index]

    @staticmethod
    def _get_index_from_tag(tag: int | Any) -> int | Any:
        """Returns the index corresponding to the node tag.

        Args:
            tag: Node tag.
        """
        return np.int64(tag - 1)

    def _init_neighbors(self, node_tags: np.ndarray) -> None:
        """Initializes the neighbors.

        Args:
            node_tags: Array of neighboring node tags. The second dimension
              is 3 for triangles and 4 for tetrahedra.
        """
        for neighbor_tags in node_tags:
            for node_index, node_tag in enumerate(neighbor_tags):
                node_array_index = self._get_index_from_tag(node_tag)
                neighbors_without_node = np.delete(neighbor_tags, node_index)
                self.neighbors[node_array_index].append(neighbors_without_node)
