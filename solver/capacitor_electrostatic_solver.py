"""The capacitor electrostatic solver solves for the electrostatic field around
a capacitor.
"""

import numpy as np
from proto.capacitor_pb2 import CapacitorEntity

from mesh.gmsh_interface import GmshNodeType
from model.capacitor import Capacitor
from model.material import MATERIAL_TO_PROPERTIES, MaterialProperties
from solver.constants import VACUUM_PERMITTIVITY
from solver.electrostatic_solver import ElectrostaticSolver2D


class CapacitorElectrostaticSolver2D(Capacitor, ElectrostaticSolver2D):
    """2D capacitor electrostatic solver."""

    def __init__(self, mesh_file: str, dc_voltage: float = 1) -> None:
        Capacitor.__init__(self, dc_voltage)
        ElectrostaticSolver2D.__init__(self, mesh_file)

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
                    self.get_material_for_entity(dim=self.dimension(),
                                                 tag=conductor_entity_tag)):
                raise ValueError(
                    f"Entity {conductor_entity_tag} is not a conductor.")
        if not MaterialProperties.is_insulator(
                self.get_material_for_entity(dim=self.dimension(),
                                             tag=CapacitorEntity.DIELECTRIC)):
            raise ValueError(
                f"Entity {CapacitorEntity.DIELECTRIC} is not an insulator.")

    def calculate_capacitance(self) -> float:
        """Calculates the capacitance."""
        ground_plate_boundary_node_tags = self.get_nodes(
            tag=CapacitorEntity.GROUND_PLATE,
            dim=self.dimension(),
            node_type=GmshNodeType.BOUNDARY)

        # Find the center of all the boundary nodes.
        node_tags, node_coordinates = self.get_node_coordinates(
            tag=CapacitorEntity.GROUND_PLATE,
            dim=self.dimension(),
            coordinates_dim=self.dimension())
        is_boundary_node_tag = np.isin(node_tags,
                                       ground_plate_boundary_node_tags)
        ground_plate_boundary_node_tags = node_tags[is_boundary_node_tag]
        ground_plate_boundary_node_coordinates = node_coordinates[
            is_boundary_node_tag]
        ground_plate_boundary_node_tag_to_coordinates = dict(
            zip(ground_plate_boundary_node_tags,
                ground_plate_boundary_node_coordinates))
        ground_plate_center = np.mean(ground_plate_boundary_node_coordinates,
                                      axis=0)

        # Find all lines along the boundary of the ground plate.
        _, line_tags = self.get_adjacencies(dim=self.dimension(),
                                            tag=CapacitorEntity.GROUND_PLATE)
        line_node_tags = [self.get_lines(tag)[1] for tag in line_tags]
        boundary_line_node_tags = np.vstack(line_node_tags)

        # Iterate over all lines along the boundary of the ground plate to find
        # the electric flux.
        total_line_length = 0
        electric_flux = 0
        for neighbor1, neighbor2 in boundary_line_node_tags:
            # Find the normal vector pointing out of the ground plate.
            # TODO(titan): Using the center of the ground plate to determine
            # the direction of the normal vector only works for convex surfaces.
            node_coordinates_neighbor1 = (
                ground_plate_boundary_node_tag_to_coordinates[neighbor1])
            node_coordinates_neighbor2 = (
                ground_plate_boundary_node_tag_to_coordinates[neighbor2])
            line_vector = (node_coordinates_neighbor2 -
                           node_coordinates_neighbor1)
            neighbor1_to_center = (ground_plate_center -
                                   node_coordinates_neighbor1)
            cross_product = np.cross(line_vector, neighbor1_to_center)
            normal_vector = (-np.sign(cross_product) *
                             np.array([[0, -1], [1, 0]]) @ line_vector)
            normal_vector /= np.linalg.norm(normal_vector)

            # Average the electric fields at the adjacent vertices.
            electric_field_neighbor1 = (
                self.electric_field[self._get_index_from_tag(neighbor1)])
            electric_field_neighbor2 = (
                self.electric_field[self._get_index_from_tag(neighbor2)])
            electric_field_averaged = (
                (electric_field_neighbor1 + electric_field_neighbor2) / 2)

            # Integrate the dot product between the electric field and the
            # normal vector.
            line_length = np.linalg.norm(line_vector)
            total_line_length += line_length
            electric_flux += (line_length *
                              np.dot(electric_field_averaged, normal_vector))
        electric_flux /= total_line_length

        # Calculate the surface charge and the capacitance.
        material = self.get_material_for_entity(dim=self.dimension(),
                                                tag=CapacitorEntity.DIELECTRIC)
        material_properties = MATERIAL_TO_PROPERTIES[material]
        Q = electric_flux * VACUUM_PERMITTIVITY * material_properties.relative_permittivity
        C = Q / self.dc_voltage
        return C
