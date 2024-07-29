"""The capacitor electrodynamic solver solves for the electromagnetic field
around a capacitor.
"""

import numpy as np
from proto.capacitor_pb2 import CapacitorEntity
from proto.solver_config_pb2 import SolverConfig

from model.py.material import MaterialProperties
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
            if not MaterialProperties.is_conductor(
                    self.get_material_for_entity(tag=conductor_entity_tag)):
                raise ValueError(
                    f"Entity {conductor_entity_tag} is not a conductor.")
        if not MaterialProperties.is_insulator(
                self.get_material_for_entity(tag=CapacitorEntity.DIELECTRIC)):
            raise ValueError(
                f"Entity {CapacitorEntity.DIELECTRIC} is not an insulator.")
