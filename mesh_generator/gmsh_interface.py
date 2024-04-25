"""The Gmsh interface class provides an interface to gmsh utilities."""

from abc import ABC
from enum import Enum, IntEnum, auto

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


class GmshNodeType(Enum):
    """Node type enumeration."""
    ALL = auto()
    INTERNAL = auto()
    BOUNDARY = auto()


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
    def get_nodes(tag: int = -1,
                  dim: int = -1,
                  node_type: GmshNodeType = GmshNodeType.ALL) -> np.ndarray:
        """Returns the list of nodes for the tag and dimension.

        Args:
            tag: Tag of the entity.
            dim: Dimension of the nodes.
            node_type: The type of nodes to return.

        Returns:
            The list of node tags. The node tags may be duplicated across multiple
            dimensions.

        Raises:
            ValueError: If the node type is invalid.
        """

        node_tags, _, _ = gmsh.model.mesh.getNodes(tag=tag,
                                                   dim=dim,
                                                   includeBoundary=True)
        internal_node_tags, _, _ = gmsh.model.mesh.getNodes(
            tag=tag, dim=dim, includeBoundary=False)
        if node_type == GmshNodeType.ALL:
            return node_tags
        if node_type == GmshNodeType.INTERNAL:
            return internal_node_tags
        if node_type == GmshNodeType.BOUNDARY:
            return np.setdiff1d(node_tags, internal_node_tags)
        raise ValueError("Invalid node type.")

    @staticmethod
    def get_node_coordinates(
            tag: int = -1,
            dim: int = -1,
            coordinates_dim: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Returns the coordinates of the nodes.

        Args:
            tag: Tag of the entity.
            dim: Dimension of the nodes.
            coordinates_dim: Dimension of the coordinates.

        Returns:
            A 2-tuple consisting of the list of node tags and the list of
            corresponding coordinates.
        """
        node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes(
            tag=tag, dim=dim, includeBoundary=True)
        return node_tags, np.reshape(node_coordinates,
                                     (-1, 3))[:, :coordinates_dim]

    @staticmethod
    def get_boundary(dim: int, tag: int) -> list[tuple[int, int]]:
        """Returns the boundary entities of the given entity.

        Args:
            dim: Dimension of the entity.
            tag: Tag of the entity.

        Returns:
            A list of 2-tuples consisting of the dimension and tag of each
            bounding entity.
        """
        return gmsh.model.getBoundary(dimTags=[(dim, tag)], oriented=False)

    @staticmethod
    def get_adjacencies(dim: int, tag: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the upward and downward adjacencies of the given entity.

        Args:
            dim: Dimension of the entity.
            tag: Tag of the entity.

        Returns:
            A 2-tuple consisting of the upward adjacent entities and the
            downward adjacent entities.
        """
        return gmsh.model.getAdjacencies(dim=dim, tag=tag)

    @staticmethod
    def get_lines(tag: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Returns the list of lines in the mesh.

        Args:
            tag: Tag of the entity.

        Returns:
            A 2-tuple consisting of the list of tags corresponding to the lines
            and the list of nodes adjacent to each line.
        """
        element_tags, node_tags = gmsh.model.mesh.getElementsByType(
            GmshElementType.TWO_NODE_LINE, tag=tag)
        return element_tags, np.reshape(node_tags, (-1, 2))

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
