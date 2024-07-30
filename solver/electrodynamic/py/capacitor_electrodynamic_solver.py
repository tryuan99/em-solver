"""The capacitor electrodynamic solver solves for the electromagnetic field
around a capacitor.
"""

import numpy as np
from proto.capacitor_pb2 import CapacitorEntity
from proto.solver_config_pb2 import SolverConfig

from model.py.material import MATERIAL_TO_PROPERTIES
from solver.electrodynamic.py.electrodynamic_solver import \
    ElectrodynamicSolver2D


class CapacitorElectrodynamicSolver2D(ElectrodynamicSolver2D):
    """2D capacitor electrodynamic solver."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__(mesh_file, solver_config)

    def _validate_mesh(self) -> None:
        """Validates the mesh.

        Raises:
            ValueError: If the mesh is invalid and cannot be solved.
        """
        # Validate the materials of the capacitor plates and the dielectric.
        for conductor_entity_tag in [
                CapacitorEntity.GROUND_PLATE,
                CapacitorEntity.VDD_PLATE,
        ]:
            if not MATERIAL_TO_PROPERTIES[self.get_material_for_entity(
                    tag=conductor_entity_tag)].is_conductor():
                raise ValueError(
                    f"Entity {conductor_entity_tag} is not a conductor.")
        if not MATERIAL_TO_PROPERTIES[self.get_material_for_entity(
                tag=CapacitorEntity.DIELECTRIC)].is_insulator():
            raise ValueError(
                f"Entity {CapacitorEntity.DIELECTRIC} is not an insulator.")
