"""The capacitor mesh generator class generates a mesh for a 2D or 3D capacitor
structure, including a parallel plate capacitor or a metal finger capacitor.
"""

from abc import ABC

import gmsh
import numpy as np

from mesh_generator.mesh_generator import (MeshGenerator, MeshGenerator2D,
                                           MeshGenerator3D)
from model.capacitor import CapacitorEntityTag


class CapacitorMeshGenerator(MeshGenerator):
    """Interface for a capacitor mesh generator."""

    def __init__(self, input_file: str) -> None:
        super().__init__(input_file)

        # Validate the model.
        self._validate_model()

    def _validate_model(self) -> None:
        """Validates the capacitor model.

        Raises:
            ValueError: If the capacitor model is invalid.
        """
        entities = self.get_entities()
        if len(entities) != 3:
            raise ValueError("Invalid number of entities.")

        # Check that tag 3 corresponds to the bounding box.
        (x_min, y_min, z_min, x_max, y_max,
         z_max) = gmsh.model.occ.getBoundingBox(
             dim=self.dimension(), tag=CapacitorEntityTag.BOUNDING_BOX_TAG)
        min_coordinates = np.array([x_min, y_min, z_min])
        max_coordinates = np.array([x_max, y_max, z_max])
        (bounding_box_min_coordinates,
         bounding_box_max_coordinates) = self.get_bounding_box()
        if not np.allclose(min_coordinates,
                           bounding_box_min_coordinates) or not np.allclose(
                               max_coordinates, bounding_box_max_coordinates):
            raise ValueError(
                f"Tag {CapacitorEntityTag.BOUNDING_BOX_TAG} does not "
                f"correspond to the bounding box.")


class CapacitorMeshGenerator2D(CapacitorMeshGenerator, MeshGenerator2D):
    """2D capacitor mesh generator."""

    pass


class CapacitorMeshGenerator3D(CapacitorMeshGenerator, MeshGenerator3D):
    """3D capacitor mesh generator."""

    pass
