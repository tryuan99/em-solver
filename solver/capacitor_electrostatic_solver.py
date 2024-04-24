"""The capacitor electrostatic solver solves for the electrostatic field around
a capacitor.
"""

import numpy as np
import scipy.sparse

from mesh_generator.gmsh_interface import GmshNodeType
from model.capacitor import Capacitor, CapacitorEntityTag
from solver.electrostatic_solver import ElectrostaticSolver2D
from solver.neighbor_lookup import NeighborLookup


class CapacitorElectrostaticSolver2D(ElectrostaticSolver2D, Capacitor):
    """2D capacitor electrostatic solver.

    In the matrix-vector equation, the first unknowns correspond to the nodes'
    voltages, and the remaining unknowns correspond to the nodes' electric
    fields.
    Similarly, the first equations correspond to Poisson's equation and any
    voltage boundary conditions, and the remaining equations correspond to the
    electric field equations for each node in the x and y-directions.
    """

    def __init__(self, mesh_file: str, dc_voltage: float = 1) -> None:
        super().__init__(mesh_file)
        Capacitor.__init__(self, dc_voltage)

    def _solve(self) -> None:
        """Implementation for solving for the voltage, the electric field, the
        magnetic vector potential, and the magnetic field.
        """
        # Get the node coordinates.
        node_tags, node_coordinates = self.get_node_coordinates(
            dim=self.dimension())
        node_tag_to_coordinates = dict(zip(node_tags, node_coordinates))

        # Initialize the matrix-vector equation.
        A = scipy.sparse.lil_matrix((self.num_unknowns, self.num_unknowns),
                                    dtype=np.float64)
        b = np.zeros(self.num_unknowns)
        equation_index = 0

        # Fill in Poisson's equations and the electric field equations for the
        # nodes within the dielectric, including the boundary nodes.
        _, dielectric_triangle_node_tags = self.get_faces(
            tag=CapacitorEntityTag.DIELECTRIC_TAG)
        dielectric_triangle_neighbors = NeighborLookup(
            dielectric_triangle_node_tags)
        dielectric_node_tags = self.get_nodes(
            tag=CapacitorEntityTag.DIELECTRIC_TAG, dim=self.dimension())
        for tag in dielectric_node_tags:
            x, y = node_tag_to_coordinates[tag]
            poisson_equation_index = (
                self._get_poisson_voltage_equation_index(tag))
            electric_field_x_equation_index = (
                self._get_electric_field_x_equation_index(tag))
            electric_field_y_equation_index = (
                self._get_electric_field_y_equation_index(tag))
            voltage_unknown_index = self._get_voltage_unknown_index(tag)
            electric_field_x_unknown_index = (
                self._get_electric_field_x_unknown_index(tag))
            electric_field_y_unknown_index = (
                self._get_electric_field_y_unknown_index(tag))
            num_adjacent_triangles = (
                dielectric_triangle_neighbors.get_num_adjacent_entities(tag))

            # Iterate through all adjacent triangles.
            for (tag_neighbor1, tag_neighbor2
                ) in dielectric_triangle_neighbors.get_neighbors(tag):
                x_neighbor1, y_neighbor1 = node_tag_to_coordinates[
                    tag_neighbor1]
                x_neighbor2, y_neighbor2 = node_tag_to_coordinates[
                    tag_neighbor2]
                voltage_unknown_index_neighbor1 = (
                    self._get_voltage_unknown_index(tag_neighbor1))
                voltage_unknown_index_neighbor2 = (
                    self._get_voltage_unknown_index(tag_neighbor2))
                electric_field_x_unknown_index_neighbor1 = (
                    self._get_electric_field_x_unknown_index(tag_neighbor1))
                electric_field_y_unknown_index_neighbor1 = (
                    self._get_electric_field_y_unknown_index(tag_neighbor1))
                electric_field_x_unknown_index_neighbor2 = (
                    self._get_electric_field_x_unknown_index(tag_neighbor2))
                electric_field_y_unknown_index_neighbor2 = (
                    self._get_electric_field_y_unknown_index(tag_neighbor2))

                # Calculate the denominator.
                denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                               x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                               x * y_neighbor1 - x * y_neighbor2)

                # Add the coefficients of the electric fields for Poisson's equation.
                A[poisson_equation_index,
                  electric_field_x_unknown_index_neighbor1] += (
                      (y_neighbor2 - y) / denominator)
                A[poisson_equation_index,
                  electric_field_y_unknown_index_neighbor1] += (
                      (x - x_neighbor2) / denominator)
                A[poisson_equation_index,
                  electric_field_x_unknown_index_neighbor2] += (
                      (y - y_neighbor1) / denominator)
                A[poisson_equation_index,
                  electric_field_y_unknown_index_neighbor2] += (
                      (x_neighbor1 - x) / denominator)
                A[poisson_equation_index, electric_field_x_unknown_index] += (
                    (y_neighbor1 - y_neighbor2) / denominator)
                A[poisson_equation_index, electric_field_y_unknown_index] += (
                    (x_neighbor2 - x_neighbor1) / denominator)

                # Add the coefficients of the voltages for the electric field
                # equation in the x-direction.
                A[electric_field_x_equation_index,
                  voltage_unknown_index_neighbor1] += ((y_neighbor2 - y) /
                                                       denominator)
                A[electric_field_x_equation_index,
                  voltage_unknown_index_neighbor2] += ((y - y_neighbor1) /
                                                       denominator)
                A[electric_field_x_equation_index,
                  voltage_unknown_index] += ((y_neighbor1 - y_neighbor2) /
                                             denominator)

                # Add the coefficients of the voltages for the electric field
                # equation in the y-direction.
                A[electric_field_y_equation_index,
                  voltage_unknown_index_neighbor1] += ((x - x_neighbor2) /
                                                       denominator)
                A[electric_field_y_equation_index,
                  voltage_unknown_index_neighbor2] += ((x_neighbor1 - x) /
                                                       denominator)
                A[electric_field_y_equation_index,
                  voltage_unknown_index] += ((x_neighbor2 - x_neighbor1) /
                                             denominator)

            # Set the coefficient for the electric fields for the electric field equations.
            A[electric_field_x_equation_index,
              electric_field_x_unknown_index] = -num_adjacent_triangles
            A[electric_field_y_equation_index,
              electric_field_y_unknown_index] = -num_adjacent_triangles

        # Fill in the voltage and electric field equations for the nodes within
        # the capacitor plates.
        for entity_tag in [
                CapacitorEntityTag.GROUND_PLATE_TAG,
                CapacitorEntityTag.VDD_PLATE_TAG
        ]:
            # Set the voltage boundary conditions and electric fields within
            # the capacitor plates.
            internal_node_tags = self.get_nodes(tag=entity_tag,
                                                dim=self.dimension(),
                                                node_type=GmshNodeType.INTERNAL)
            for tag in internal_node_tags:
                voltage_unknown_index = self._get_voltage_unknown_index(tag)
                electric_field_x_unknown_index = (
                    self._get_electric_field_x_unknown_index(tag))
                electric_field_y_unknown_index = (
                    self._get_electric_field_y_unknown_index(tag))

                # Voltage boundary conditions.
                voltage_equation_index = (
                    self._get_poisson_voltage_equation_index(tag))
                A[voltage_equation_index, voltage_unknown_index] = 1
                b[voltage_equation_index] = self.get_dc_voltage(entity_tag)

                # Electric field in the x-direction.
                electric_field_x_equation_index = (
                    self._get_electric_field_x_equation_index(tag))
                A[electric_field_x_equation_index,
                  electric_field_x_unknown_index] = 1

                # Electric field in the y-direction.
                electric_field_y_equation_index = (
                    self._get_electric_field_y_equation_index(tag))
                A[electric_field_y_equation_index,
                  electric_field_y_unknown_index] = 1

            # Set the voltage boundary conditions for the boundary nodes at the
            # capacitor plates.
            boundary_node_tags = self.get_nodes(tag=entity_tag,
                                                dim=self.dimension(),
                                                node_type=GmshNodeType.BOUNDARY)
            for tag in boundary_node_tags:
                voltage_unknown_index = self._get_voltage_unknown_index(tag)

                # Voltage boundary conditions.
                voltage_equation_index = (
                    self._get_poisson_voltage_equation_index(tag))
                A[voltage_equation_index, :] = 0
                A[voltage_equation_index, voltage_unknown_index] = 1
                b[voltage_equation_index] = self.get_dc_voltage(entity_tag)

        # Solve for the voltage and the electric field.
        x = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        self.voltage = x[:self.num_voltage_unknowns]
        self.electric_field = np.reshape(
            x[self.num_voltage_unknowns:self.num_voltage_unknowns +
              self.num_electric_field_unknowns], (-1, self.dimension()))

    def _get_voltage_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's voltage."""
        return self._get_index_from_tag(tag)

    def _get_electric_field_x_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the x-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + self.num_voltage_unknowns

    def _get_electric_field_y_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the y-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + 1 + self.num_voltage_unknowns

    def _get_poisson_voltage_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Poisson's equation or
        any voltage boundary condition.
        """
        return self._get_index_from_tag(tag)

    def _get_electric_field_x_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to the node's electric field
        in the x-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + self.num_voltage_unknowns

    def _get_electric_field_y_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to the node's electric field
        in the y-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + 1 + self.num_voltage_unknowns
