"""The mesh generator class generates a mesh for a 2D or 3D structure."""

from abc import ABC, abstractmethod
from enum import IntEnum

import gmsh
import numpy as np

from mesh_generator.gmsh_interface import GmshInterface

# Mesh resolution factor. The higher the factor, the finer the resolution.
MESH_RESOLUTION_FACTOR = 100

# Mesh bounding box factor.
MESH_BOUNDING_BOX_FACTOR = 2

# Mesh threshold field.
MESH_THRESHOLD_FIELD_LC_MIN_FACTOR = 1
MESH_THRESHOLD_FIELD_LC_MAX_FACTOR = 10
MESH_THRESHOLD_FIELD_DISTANCE_MIN_FACTOR = 1
MESH_THRESHOLD_FIELD_DISTANCE_MAX_FACTOR = 10


class MeshGenerator(GmshInterface):
    """Interface for a mesh generator.

    Args:
        dimension: Dimension of the structure and the generated mesh.
    """

    def __init__(self, input_file: str) -> None:
        super().__init__()

        # Generate the mesh.
        self._generate_mesh(input_file)

    def __del__(self) -> None:
        gmsh.finalize()

    @classmethod
    @abstractmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""

    @abstractmethod
    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the bounding box of the entire structure.

        Returns:
            A 2-tuple consisting of the minimum (x, y, z) coordinates and the
            maximum (x, y, z) coordinates.
        """

    def get_entities(self) -> list[tuple[int, int]]:
        """Returns the structures in the mesh.

        Returns:
            A list of 2-tuples consisting of the dimension and tag of each
            structure.
        """
        return gmsh.model.getEntities(dim=self.dimension())

    def _generate_mesh(self, input_file: str) -> None:
        """Generates a mesh.

        Args:
            input_file: Input file.
        """
        # Open the input file.
        gmsh.open(input_file)
        gmsh.model.occ.synchronize()
        entities = self.get_entities()

        # Calculate the dimensions.
        min_coordinates, max_coordinates = self.get_bounding_box()
        dimensions = max_coordinates - min_coordinates

        # Add a bounding box.
        bounding_box_tag = self._add_bounding_box(min_coordinates, dimensions)

        # Remove the structure from the bounding box.
        dielectric, _ = gmsh.model.occ.cut(
            [(self.dimension(), bounding_box_tag)],
            entities,
            removeObject=True,
            removeTool=False)

        # Add a mesh field as a function of the distance to the structure.
        gmsh.model.occ.synchronize()
        self._add_mesh_field(entities, dimensions)

        # Generate a mesh.
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(self.dimension())

    def _add_bounding_box(
            self,
            min_coordinates: np.ndarray,
            dimensions: np.ndarray,
            bounding_box_factor: float = MESH_BOUNDING_BOX_FACTOR) -> int:
        """Adds a bounding box to the model.

        Args:
            min_coordinates: Minimum (x, y, z) coordinates of the structure.
            dimensions: The (x, y, z) dimensions of the structure.
            bounding_box_factor: Ratio of the additional bounding box dimension
              to the minimum structure dimension.

        Returns:
            The tag of the bounding box.
        """
        minimum_dimension = np.min(dimensions[:self.dimension()])
        box_dimension = minimum_dimension * bounding_box_factor
        box_min_coordinates = min_coordinates - box_dimension * (np.arange(
            len(min_coordinates)) < self.dimension())
        box_dimensions = dimensions + 2 * box_dimension
        return self._add_bounding_box_impl(box_min_coordinates,
                                           box_dimensions[:self.dimension()])

    @classmethod
    @abstractmethod
    def _add_bounding_box_impl(cls, min_coordinates: np.ndarray,
                               dimensions: np.ndarray):
        """Adds a bounding to the model.

        Args:
            min_coordinates: Minimum (x, y, z) coordinates of the structure.
            dimensions: The dimensions of the structure.

        Returns:
            The tag of the bounding box.
        """

    def _add_mesh_field(self, entities: list[tuple[int, int]],
                        dimensions: int) -> None:
        """Adds a mesh field as a function of the distance to the structure.

        Args:
            entities: Structure entities.
            dimensions: The (x, y, z) dimensions of the structure.
        """
        boundaries = gmsh.model.getBoundary(dimTags=entities, oriented=False)
        boundary_tags = [boundary[1] for boundary in boundaries]

        # Add a distance field.
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "FacesList", boundary_tags)

        # Add a threshold field.
        resolution = np.min(
            dimensions[:self.dimension()]) / MESH_RESOLUTION_FACTOR
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
        gmsh.model.mesh.field.setNumber(
            threshold, "LcMin", MESH_THRESHOLD_FIELD_LC_MIN_FACTOR * resolution)
        gmsh.model.mesh.field.setNumber(
            threshold, "LcMax", MESH_THRESHOLD_FIELD_LC_MAX_FACTOR * resolution)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMin",
            MESH_THRESHOLD_FIELD_DISTANCE_MIN_FACTOR * resolution)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMax",
            MESH_THRESHOLD_FIELD_DISTANCE_MAX_FACTOR * resolution)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)


class MeshGenerator2D(MeshGenerator):
    """2D mesh generator."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 2

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the bounding box of the entire structure.

        Returns:
            A 2-tuple consisting of the minimum (x, y, z) coordinates and the
            maximum (x, y, z) coordinates.
        """
        # Find the minimum and maximum coordinates of all entities.
        entities = self.get_entities()
        entity_bounding_boxes = np.array([
            gmsh.model.occ.getBoundingBox(dim=dimension, tag=tag)
            for dimension, tag in entities
        ])
        min_entity_coordinates = entity_bounding_boxes[:, :3]
        min_coordinates = np.min(min_entity_coordinates, axis=0)
        max_entity_coordinates = entity_bounding_boxes[:, 3:]
        max_coordinates = np.max(max_entity_coordinates, axis=0)
        return min_coordinates, max_coordinates

    @classmethod
    def _add_bounding_box_impl(cls, min_coordinates: np.ndarray,
                               dimensions: np.ndarray):
        """Adds a bounding to the model.

        Args:
            min_coordinates: Minimum (x, y, z) coordinates of the structure.
            dimensions: The (x, y) dimensions of the structure.

        Returns:
            The tag of the bounding box.
        """
        return gmsh.model.occ.addRectangle(*min_coordinates, *dimensions)


class MeshGenerator3D(MeshGenerator):
    """3D mesh generator."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the bounding box of the entire structure.

        Returns:
            A 2-tuple consisting of the minimum (x, y, z) coordinates and the
            maximum (x, y, z) coordinates.
        """
        (x_min, y_min, z_min, x_max, y_max,
         z_max) = gmsh.model.getBoundingBox(dim=-1, tag=-1)
        return np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])

    @classmethod
    def _add_bounding_box_impl(cls, min_coordinates: np.ndarray,
                               dimensions: np.ndarray):
        """Adds a bounding to the model.

        Args:
            min_coordinates: Minimum (x, y, z) coordinates of the structure.
            dimensions: The (x, y, z) dimensions of the structure.

        Returns:
            The tag of the bounding box.
        """
        return gmsh.model.occ.addBox(*min_coordinates, *dimensions)
