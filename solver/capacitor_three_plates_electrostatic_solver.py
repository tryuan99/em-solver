"""The capacitor three plates electrostatic solver solves for the electrostatic
field around a capacitor with three plates.
"""

from proto.capacitor_three_plates_pb2 import CapacitorThreePlatesEntity

from model.capacitor_three_plates import CapacitorThreePlates
from model.material import MATERIAL_TO_PROPERTIES, MaterialProperties
from solver.electrostatic_solver import ElectrostaticSolver2D


class CapacitorThreePlatesElectrostaticSolver2D(CapacitorThreePlates,
                                                ElectrostaticSolver2D):
    """2D capacitor electrostatic solver with three plates."""

    def __init__(self, mesh_file: str, dc_voltage: float = 1) -> None:
        CapacitorThreePlates.__init__(self, dc_voltage)
        ElectrostaticSolver2D.__init__(self, mesh_file)

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
            if not MaterialProperties.is_conductor(
                    self.get_material_for_entity(dim=self.dimension(),
                                                 tag=conductor_entity_tag)):
                raise ValueError(
                    f"Entity {conductor_entity_tag} is not a conductor.")
        if not MaterialProperties.is_insulator(
                self.get_material_for_entity(
                    dim=self.dimension(),
                    tag=CapacitorThreePlatesEntity.DIELECTRIC)):
            raise ValueError(
                f"Entity {CapacitorThreePlatesEntity.DIELECTRIC} is not an insulator."
            )
