"""The capacitor mesh generator class generates a mesh for a 2D or 3D capacitor
structure, including a parallel plate capacitor or a metal finger capacitor.
"""

from abc import ABC
from enum import IntEnum

import gmsh
import numpy as np

from mesh_generator.mesh_generator import (MeshGenerator, MeshGenerator2D,
                                           MeshGenerator3D)


class CapacitorEntityTag(IntEnum):
    """Capacitor entity tag enumeration."""
    GROUND_PLATE_TAG = 1
    VDD_PLATE_TAG = 2
    BOUNDING_BOX_TAG = 3


class CapacitorMeshGenerator(MeshGenerator):
    """Interface for a capacitor mesh generator.

    Attributes:
        dc_voltage: DC voltage applied to the capacitor plates.
    """

    def __init__(self, input_file: str, dc_voltage: float) -> None:
        super().__init__(input_file)
        self.dc_voltage = dc_voltage

        # Validate the model.
        self._validate_model()

    def get_dc_voltage(self, tag: int) -> float:
        """Returns the DC voltage for the tag.

        Args:
            tag: Tag of the entity.

        Raises:
            ValueError: If the tag does not belong to the capacitor structure.
        """
        if tag == CapacitorEntityTag.GROUND_PLATE_TAG:
            return 0
        if tag == CapacitorEntityTag.VDD_PLATE_TAG:
            return self.dc_voltage
        raise ValueError("Invalid capacitor entity tag.")

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

    def __init__(self, input_file: str, dc_voltage: float = -1):
        super().__init__(input_file, dc_voltage)


class CapacitorMeshGenerator3D(CapacitorMeshGenerator, MeshGenerator3D):
    """3D capacitor mesh generator."""

    def __init__(self, input_file: str, dc_voltage: float = -1):
        super().__init__(input_file, dc_voltage)
