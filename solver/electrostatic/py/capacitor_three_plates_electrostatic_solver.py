"""The capacitor three plates electrostatic solver solves for the electrostatic
field around a capacitor with three plates.
"""

from proto.capacitor_three_plates_pb2 import CapacitorThreePlatesEntity
from proto.solver_config_pb2 import SolverConfig

from model.py.material import MATERIAL_TO_PROPERTIES
from solver.electrostatic.py.electrostatic_solver import ElectrostaticSolver2D


class CapacitorThreePlatesElectrostaticSolver2D(ElectrostaticSolver2D):
    """2D capacitor electrostatic solver with three plates."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__(mesh_file, solver_config)

    def _validate_mesh(self) -> None:
        """Validates the mesh.

        Raises:
            ValueError: If the mesh is invalid and cannot be solved.
        """
        # Validate the materials of the capacitor plates and the dielectric.
        for conductor_entity_tag in [
                CapacitorThreePlatesEntity.GROUND_PLATE,
                CapacitorThreePlatesEntity.MIDDLE_PLATE,
                CapacitorThreePlatesEntity.VDD_PLATE,
        ]:
            if not MATERIAL_TO_PROPERTIES[self.get_material_for_entity(
                    tag=conductor_entity_tag)].is_conductor():
                raise ValueError(
                    f"Entity {conductor_entity_tag} is not a conductor.")
        if not MATERIAL_TO_PROPERTIES[self.get_material_for_entity(
                tag=CapacitorThreePlatesEntity.DIELECTRIC)].is_insulator():
            raise ValueError(
                f"Entity {CapacitorThreePlatesEntity.DIELECTRIC} is not an "
                f"insulator.")
