"""The mesh generator class generates a mesh for a 2D or 3D structure."""

from abc import ABC, abstractmethod

import gmsh
import numpy as np
from proto.material_pb2 import Material
from proto.mesh_config_pb2 import EntityConfig, MeshConfig, MeshGeneratorConfig

from mesh.py.gmsh_interface import GmshInterface

# Mesh bounding box factor.
MESH_BOUNDING_BOX_FACTOR = 2

# Linear number of sampling points per dimension for calculating the distance
# for the mesh field.
MESH_DISTANCE_NUM_SAMPLING_POINTS = 10000


class MeshGenerator(GmshInterface, ABC):
    """Interface for a mesh generator.

    Args:
        dimension: Dimension of the structure and the generated mesh.
    """

    def __init__(self, input_file: str, mesh_config: MeshConfig) -> None:
        super().__init__()

        # Generate the mesh.
        self._generate_mesh(input_file, mesh_config)

        # Validate the mesh.
        self._validate_mesh()

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

    def _generate_mesh(self, input_file: str, mesh_config: MeshConfig) -> None:
        """Generates a mesh.

        Args:
            input_file: Input file.
            mesh_config: Mesh configuration.
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
        gmsh.model.occ.cut([(self.dimension(), bounding_box_tag)],
                           entities,
                           removeObject=True,
                           removeTool=False)
        gmsh.model.occ.synchronize()

        # Define physical groups to define the material.
        self._add_physical_groups(mesh_config.entity_configs)

        # Add a mesh field as a function of the distance to the structure.
        self._add_mesh_field(mesh_config.generator_config, entities, dimensions)

        # Generate a mesh.
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(dim=self.dimension())

    def _validate_mesh(self) -> None:
        """Validates the mesh.

        Raises:
            ValueError: If the mesh is invalid.
        """
        # Validate that the entity tags are consecutive starting from 1.
        entities = self.get_entities()
        entity_tags = [entity[1] for entity in entities]
        max_entity_tag = max(entity_tags)
        if max_entity_tag != len(entities):
            raise ValueError("Entity tags are not consecutive.")

        # Check that the bounding box bounds all other entities. The maximum
        # entity tag belongs to the bounding box.
        (x_min, y_min, z_min, x_max, y_max,
         z_max) = gmsh.model.occ.getBoundingBox(dim=self.dimension(),
                                                tag=max_entity_tag)
        min_coordinates = np.array([x_min, y_min, z_min])
        max_coordinates = np.array([x_max, y_max, z_max])
        (bounding_box_min_coordinates,
         bounding_box_max_coordinates) = self.get_bounding_box()
        if not np.allclose(min_coordinates,
                           bounding_box_min_coordinates) or not np.allclose(
                               max_coordinates, bounding_box_max_coordinates):
            raise ValueError(
                f"Entity {max_entity_tag} does not correspond to the bounding "
                f"box.")

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

    def _add_physical_groups(self, entity_configs: list[EntityConfig]) -> None:
        """Adds physical groups to define the entity material.

        Args:
            entity_configs: Entity configurations.
        """
        entities = self.get_entities()
        entity_tags = [entity[1] for entity in entities]

        # Group entities of the same material into the same physical group.
        entity_tags_by_material = {}
        for entity_config in entity_configs:
            if entity_config.tag in entity_tags:
                entity_tags_by_material.setdefault(entity_config.material,
                                                   []).append(entity_config.tag)

        # Define a physical group for each material.
        for material, entity_tags in entity_tags_by_material.items():
            gmsh.model.addPhysicalGroup(dim=self.dimension(),
                                        tags=entity_tags,
                                        name=Material.Name(material))

    def _add_mesh_field(self, mesh_generator_config: MeshGeneratorConfig,
                        entities: list[tuple[int,
                                             int]], dimensions: int) -> None:
        """Adds a mesh field as a function of the distance to the structure.

        Args:
            mesh_generator_config: Mesh generator configuration.
            entities: Structure entities.
            dimensions: The (x, y, z) dimensions of the structure.
        """
        boundaries = gmsh.model.getBoundary(dimTags=entities, oriented=False)
        boundary_tags = [boundary[1] for boundary in boundaries]

        # Add a distance field.
        distance = gmsh.model.mesh.field.add("Distance")
        if self.dimension() == 2:
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList",
                                             boundary_tags)
        elif self.dimension() == 3:
            # SurfacesList only supports OpenCASCADE and discrete surfaces, so
            # use the bounding curves for the distance field instead.
            line_tags = []
            for boundary_tag in boundary_tags:
                _, boundary_line_tags = self.get_adjacencies(
                    dim=self.dimension() - 1, tag=boundary_tag)
                line_tags.extend(boundary_line_tags)
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList", line_tags)
        gmsh.model.mesh.field.setNumber(distance, "Sampling",
                                        MESH_DISTANCE_NUM_SAMPLING_POINTS)

        # Add a threshold field.
        min_dimension = np.min(dimensions[:self.dimension()])
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
        gmsh.model.mesh.field.setNumber(
            threshold, "LcMin",
            mesh_generator_config.lc_min_factor * min_dimension)
        gmsh.model.mesh.field.setNumber(
            threshold, "LcMax",
            mesh_generator_config.lc_max_factor * min_dimension)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMin",
            mesh_generator_config.distance_min_factor * min_dimension)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMax",
            mesh_generator_config.distance_max_factor * min_dimension)
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
