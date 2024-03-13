"""The Gmsh interface class provides an interface to gmsh utilities."""

from abc import ABC
from enum import IntEnum

import gmsh
import numpy as np


class GmshElementType(IntEnum):
    """Mesh element type enumeration."""
    TWO_NODE_LINE = 1
    THREE_NODE_TRIANGLE = 2
    FOUR_NODE_QUADRANGLE = 3
    FOUR_NODE_TETRAHEDRON = 4
    EIGHT_NODE_HEXAHEDRON = 5
    SIX_NODE_PRISM = 6
    FIVE_NODE_PYRAMID = 7
    ONE_NODE_POINT = 15


class GmshInterface(ABC):
    """Gmsh interface."""

    def __init__(self) -> None:
        gmsh.initialize()

    def __del__(self) -> None:
        gmsh.finalize()

    @staticmethod
    def write_mesh_file(mesh_file: str) -> None:
        """Writes the mesh to the output file.

        Args:
            mesh_file: Mesh file.
        """
        gmsh.write(mesh_file)

    @staticmethod
    def get_nodes(tag: int = -1, boundary_only: bool = False) -> np.ndarray:
        """Returns the list of nodes for the tag.

        Args:
            tag: Tag of the entity.
            boundary_only: If true, return boundary nodes only. Otherwise,
              return all nodes.

        Returns:
            The list of node tags.
        """
        node_tags, _, _ = gmsh.model.mesh.getNodes(tag=tag,
                                                   includeBoundary=True)
        if not boundary_only:
            return node_tags
        internal_node_tags, _, _ = gmsh.model.mesh.getNodes(
            tag=tag, includeBoundary=False)
        return np.setdiff1d(node_tags, internal_node_tags)

    @staticmethod
    def get_node_coordinates(tag: int = -1) -> dict[int, np.ndarray]:
        """Returns the coordinates of the nodes.

        Args:
            tag: Tag of the entity.

        Returns:
            A dictionary mapping from the node tag to the node's coordinates.
        """
        node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes(
            tag=tag, includeBoundary=True)
        return dict(zip(node_tags, node_coordinates))

    @staticmethod
    def get_faces(tag: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Returns the list of triangular faces in the mesh.

        Args:
            tag: Tag of the entity.

        Returns:
            A 2-tuple consisting of the list of tags corresponding to the faces
            and the list of nodes belonging to each face.
        """
        element_tags, node_tags = gmsh.model.mesh.getElementsByType(
            GmshElementType.THREE_NODE_TRIANGLE, tag=tag)
        return element_tags, np.reshape(node_tags, (-1, 3))

    @staticmethod
    def get_tetrahedra(tag: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Returns the list of tetrahedra volumes in the mesh.

        Args:
            tag: Tag of the entity.

        Returns:
            A 2-tuple consisting of the list of tags corresponding to the
            tetrahedra and the list of nodes belonging to each volume.
        """
        element_tags, node_tags = gmsh.model.mesh.getElementsByType(
            GmshElementType.FOUR_NODE_TETRAHEDRON, tag=tag)
        return element_tags, np.reshape(node_tags, (-1, 4))
