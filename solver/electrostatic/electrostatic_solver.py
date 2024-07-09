"""The electrostatic solver solves for the electric potential and the electric
field only.
"""

import numpy as np
import scipy.sparse
from proto.solver_config_pb2 import SolverConfig

from mesh.gmsh_interface import GmshNodeType
from model.material import MaterialProperties
from solver.electromagnetic_solver import (ElectromagneticSolver,
                                           ElectromagneticSolver2D,
                                           ElectromagneticSolver3D)
from solver.neighbor_lookup import NeighborLookup


class ElectrostaticSolver(ElectromagneticSolver):
    """Interface for an electrostatic solver."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__(mesh_file, solver_config)

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return (self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns)

    def _get_dc_voltage(self, tag: int) -> float:
        """Returns the DC voltage for the tag.

        Args:
            tag: Tag of the entity.

        Raises:
            ValueError: If the tag cannot be found.
        """
        for entity_config in self.config.entity_configs:
            if entity_config.tag == tag:
                return entity_config.dc_voltage
        raise ValueError(f"Entity {tag} cannot be found.")


class ElectrostaticSolver2D(ElectrostaticSolver, ElectromagneticSolver2D):
    """Interface for a 2D electrostatic solver."""

    def _solve(self) -> None:
        """Implementation for solving for the electric potential, the electric
        field, the magnetic vector potential, and the magnetic flux density.

        In the matrix-vector equation, the first unknowns correspond to the
        nodes' electric potentials, and the remaining unknowns correspond to
        the nodes' electric fields.
        Similarly, the first equations correspond to Poisson's equation and any
        voltage boundary conditions, and the remaining equations correspond to
        the electric field equations for each node in the x and y-directions.
        """
        # Get the node coordinates.
        node_tags, node_coordinates = self.get_node_coordinates(
            dim=self.dimension(), coordinates_dim=self.dimension())
        node_tag_to_coordinates = dict(zip(node_tags, node_coordinates))

        # Get the conductor and the insulator entities.
        physical_groups = self.get_physical_groups(dim=self.dimension())
        conductor_entity_tags = []
        insulator_entity_tags = []
        for _, physical_group_tag in physical_groups:
            material = self.get_material_for_physical_group(
                tag=physical_group_tag)
            entities = self.get_entities_for_physical_group(
                dim=self.dimension(), tag=physical_group_tag)
            if MaterialProperties.is_conductor(material):
                conductor_entity_tags.extend(entities)
            elif MaterialProperties.is_insulator(material):
                insulator_entity_tags.extend(entities)

        # Initialize the matrix-vector equation.
        A = scipy.sparse.lil_matrix((self.num_unknowns, self.num_unknowns),
                                    dtype=np.float64)
        b = np.zeros(self.num_unknowns)

        # Fill in Poisson's equations and the electric field equations for the
        # nodes within the insulators, including the boundary nodes.
        for insulator_tag in insulator_entity_tags:
            _, insulator_triangle_node_tags = self.get_faces(tag=insulator_tag)
            insulator_triangle_neighbors = NeighborLookup(
                insulator_triangle_node_tags)
            insulator_node_tags = self.get_nodes(tag=insulator_tag,
                                                 dim=self.dimension())
            for tag in insulator_node_tags:
                x, y = node_tag_to_coordinates[tag]
                poisson_equation_index = (
                    self._get_poisson_electric_potential_equation_index(tag))
                electric_field_x_equation_index = (
                    self._get_electric_field_x_equation_index(tag))
                electric_field_y_equation_index = (
                    self._get_electric_field_y_equation_index(tag))
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))
                electric_field_x_unknown_index = (
                    self._get_electric_field_x_unknown_index(tag))
                electric_field_y_unknown_index = (
                    self._get_electric_field_y_unknown_index(tag))
                num_adjacent_triangles = (
                    insulator_triangle_neighbors.get_num_adjacent_entities(tag))

                # Iterate over all adjacent triangles.
                for (tag_neighbor1, tag_neighbor2
                    ) in insulator_triangle_neighbors.get_neighbors(tag):
                    x_neighbor1, y_neighbor1 = node_tag_to_coordinates[
                        tag_neighbor1]
                    x_neighbor2, y_neighbor2 = node_tag_to_coordinates[
                        tag_neighbor2]
                    electric_potential_unknown_index_neighbor1 = (
                        self._get_electric_potential_unknown_index(
                            tag_neighbor1))
                    electric_potential_unknown_index_neighbor2 = (
                        self._get_electric_potential_unknown_index(
                            tag_neighbor2))
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

                    # Add the coefficients of the electric fields for Poisson's
                    # equation.
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
                    A[poisson_equation_index,
                      electric_field_x_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[poisson_equation_index,
                      electric_field_y_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                    # Add the coefficients of the electric potentials for the
                    # electric field equation in the x-direction.
                    A[electric_field_x_equation_index,
                      electric_potential_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[electric_field_x_equation_index,
                      electric_potential_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[electric_field_x_equation_index,
                      electric_potential_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)

                    # Add the coefficients of the electric potentials for the
                    # electric field equation in the y-direction.
                    A[electric_field_y_equation_index,
                      electric_potential_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[electric_field_y_equation_index,
                      electric_potential_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[electric_field_y_equation_index,
                      electric_potential_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                # Set the coefficient for the electric fields for the
                # electric field equations.
                A[electric_field_x_equation_index,
                  electric_field_x_unknown_index] = num_adjacent_triangles
                A[electric_field_y_equation_index,
                  electric_field_y_unknown_index] = num_adjacent_triangles

        # Fill in the electric potential and electric field equations for the
        # nodes within the conductors, revising the equations for the boundary
        # nodes.
        # At steady state, the electric potential is constant throughout the
        # insulator, and the electric field is zero throughout, even with a
        # non-zero resistivity.
        for conductor_tag in conductor_entity_tags:
            # Set the voltage boundary conditions and electric fields within
            # the capacitor plates.
            conductor_internal_node_tags = self.get_nodes(
                tag=conductor_tag,
                dim=self.dimension(),
                node_type=GmshNodeType.INTERNAL)
            for tag in conductor_internal_node_tags:
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))
                electric_field_x_unknown_index = (
                    self._get_electric_field_x_unknown_index(tag))
                electric_field_y_unknown_index = (
                    self._get_electric_field_y_unknown_index(tag))

                # Set the voltage boundary conditions.
                electric_potential_equation_index = (
                    self._get_poisson_electric_potential_equation_index(tag))
                A[electric_potential_equation_index,
                  electric_potential_unknown_index] = 1
                b[electric_potential_equation_index] = (
                    self._get_dc_voltage(conductor_tag))

                # Set the electric field boundary conditions in the x-direction.
                electric_field_x_equation_index = (
                    self._get_electric_field_x_equation_index(tag))
                A[electric_field_x_equation_index,
                  electric_field_x_unknown_index] = 1

                # Set the electric field boundary conditions in the y-direction.
                electric_field_y_equation_index = (
                    self._get_electric_field_y_equation_index(tag))
                A[electric_field_y_equation_index,
                  electric_field_y_unknown_index] = 1

            # Set the voltage boundary conditions for the boundary nodes at the
            # capacitor plates.
            conductor_boundary_node_tags = self.get_nodes(
                tag=conductor_tag,
                dim=self.dimension(),
                node_type=GmshNodeType.BOUNDARY)
            for tag in conductor_boundary_node_tags:
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))

                # Set the voltage boundary conditions.
                electric_potential_equation_index = (
                    self._get_poisson_electric_potential_equation_index(tag))
                A[electric_potential_equation_index, :] = 0
                A[electric_potential_equation_index,
                  electric_potential_unknown_index] = 1
                b[electric_potential_equation_index] = (
                    self._get_dc_voltage(conductor_tag))

        # Solve for the electric potential and the electric field.
        x = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        self.electric_potential = x[:self.num_electric_potential_unknowns]
        self.electric_field = np.reshape(
            x[self.num_electric_potential_unknowns:self.
              num_electric_potential_unknowns +
              self.num_electric_field_unknowns], (-1, self.dimension()))

    def _get_electric_potential_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric
        potential.
        """
        return self._get_index_from_tag(tag)

    def _get_electric_field_x_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the x-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + self.num_electric_potential_unknowns

    def _get_electric_field_y_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the y-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + 1 + self.num_electric_potential_unknowns

    def _get_poisson_electric_potential_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Poisson's equation or
        any voltage boundary condition.
        """
        return self._get_index_from_tag(tag)

    def _get_electric_field_x_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to the node's electric field
        in the x-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + self.num_electric_potential_unknowns

    def _get_electric_field_y_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to the node's electric field
        in the y-direction.
        """
        return self.dimension() * self._get_index_from_tag(
            tag) + 1 + self.num_electric_potential_unknowns


class ElectrostaticSolver3D(ElectrostaticSolver, ElectromagneticSolver3D):
    """Interface for a 3D electrostatic solver."""

    pass
