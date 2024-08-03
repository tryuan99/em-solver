"""The electrodynamic solver solves for the electric potential, the electric
field, the magnetic vector potential, and the magnetic flux density at a given
frequency.
"""

import numpy as np
import scipy.sparse
from proto.solver_config_pb2 import SolverConfig

from mesh.py.gmsh_interface import GmshNodeType
from model.py.material import MATERIAL_TO_PROPERTIES
from solver.py.electromagnetic_solver import (ElectromagneticSolver,
                                              ElectromagneticSolver2D,
                                              ElectromagneticSolver3D)
from solver.py.neighbor_lookup import NeighborLookup


class ElectrodynamicSolver(ElectromagneticSolver):
    """Interface for an electrodynamic solver."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__(mesh_file, solver_config)

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return (self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_vector_potential_unknowns +
                self.num_magnetic_flux_density_unknowns + self.num_nodes)

    def _get_ac_phasor(self, tag: int) -> np.complex128:
        """Returns the AC phasor voltage for the tag.

        Args:
            tag: Tag of the entity.

        Raises:
            ValueError: If the tag cannot be found.
        """
        for entity_config in self.config.entity_configs:
            if entity_config.tag == tag:
                return (entity_config.ac_phasor.real +
                        1j * entity_config.ac_phasor.imaginary)
        raise ValueError(f"Entity {tag} cannot be found.")


class ElectrodynamicSolver2D(ElectrodynamicSolver, ElectromagneticSolver2D):
    """Interface for a 2D electrodynamic solver."""

    def _solve(self) -> None:
        """Implementation for solving for the electric potential, the electric
        field, the magnetic vector potential, and the magnetic flux density.

        In the matrix-vector equation, the first unknowns correspond to the
        nodes' electric potentials, the next unknowns correspond to the nodes'
        electric fields, and the next unknowns correspond to the nodes'
        magnetic vector potentials, the next unknowns correspond to the nodes'
        magnetic flux densities, and the last unknowns correspond to the
        Lagrange multiplier for the gauge.
        Similarly, the first equations correspond to Gauss's law and voltage
        boundary conditions, the next equations correspond to Faraday's law,
        the next equations correspond to Gauss's law for magnetism, the
        next equations correspond to Ampere's law, and the last equation
        corresponds to the Lagrange multiplier for the gauge.
        """
        omega = 2 * np.pi * self.config.frequency

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
            if MATERIAL_TO_PROPERTIES[material].is_conductor():
                conductor_entity_tags.extend(entities)
            elif MATERIAL_TO_PROPERTIES[material].is_insulator():
                insulator_entity_tags.extend(entities)

        # Initialize the matrix-vector equation.
        A = scipy.sparse.lil_matrix((self.num_unknowns, self.num_unknowns),
                                    dtype=np.complex128)
        b = np.zeros(self.num_unknowns, dtype=np.complex128)

        # Fill in the equations for the nodes within the insulators, including
        # the boundary nodes.
        for insulator_tag in insulator_entity_tags:
            insulator_properties = MATERIAL_TO_PROPERTIES[
                self.get_material_for_entity(insulator_tag)]
            _, insulator_triangle_node_tags = self.get_faces(tag=insulator_tag)
            insulator_triangle_neighbors = NeighborLookup(
                insulator_triangle_node_tags)
            insulator_node_tags = self.get_nodes(tag=insulator_tag,
                                                 dim=self.dimension())
            for tag in insulator_node_tags:
                x, y = node_tag_to_coordinates[tag]
                num_adjacent_triangles = (
                    insulator_triangle_neighbors.get_num_adjacent_entities(tag))

                # Equation indices.
                gauss_law_electric_potential_equation_index = (
                    self._get_gauss_law_electric_potential_equation_index(tag))
                faraday_law_x_equation_index = (
                    self._get_faraday_law_x_equation_index(tag))
                faraday_law_y_equation_index = (
                    self._get_faraday_law_y_equation_index(tag))
                gauss_law_magnetism_z_equation_index = (
                    self._get_gauss_law_magnetism_z_equation_index(tag))
                ampere_law_x_equation_index = (
                    self._get_ampere_law_x_equation_index(tag))
                ampere_law_y_equation_index = (
                    self._get_ampere_law_y_equation_index(tag))
                gauge_equation_index = self._get_gauge_equation_index(tag)

                # Unknown indices.
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))
                electric_field_x_unknown_index = (
                    self._get_electric_field_x_unknown_index(tag))
                electric_field_y_unknown_index = (
                    self._get_electric_field_y_unknown_index(tag))
                magnetic_vector_potential_x_unknown_index = (
                    self._get_magnetic_vector_potential_x_unknown_index(tag))
                magnetic_vector_potential_y_unknown_index = (
                    self._get_magnetic_vector_potential_y_unknown_index(tag))
                magnetic_flux_density_z_unknown_index = (
                    self._get_magnetic_flux_density_z_unknown_index(tag))
                gauge_unknown_index = self._get_gauge_unknown_index(tag)

                # Iterate over all adjacent triangles.
                for (tag_neighbor1, tag_neighbor2
                    ) in insulator_triangle_neighbors.get_neighbors(tag):
                    x_neighbor1, y_neighbor1 = node_tag_to_coordinates[
                        tag_neighbor1]
                    x_neighbor2, y_neighbor2 = node_tag_to_coordinates[
                        tag_neighbor2]

                    # Unknown indices for both neighbors.
                    electric_potential_unknown_index_neighbor1 = (
                        self._get_electric_potential_unknown_index(
                            tag_neighbor1))
                    electric_potential_unknown_index_neighbor2 = (
                        self._get_electric_potential_unknown_index(
                            tag_neighbor2))
                    electric_field_x_unknown_index_neighbor1 = (
                        self._get_electric_field_x_unknown_index(tag_neighbor1))
                    electric_field_x_unknown_index_neighbor2 = (
                        self._get_electric_field_x_unknown_index(tag_neighbor2))
                    electric_field_y_unknown_index_neighbor1 = (
                        self._get_electric_field_y_unknown_index(tag_neighbor1))
                    electric_field_y_unknown_index_neighbor2 = (
                        self._get_electric_field_y_unknown_index(tag_neighbor2))
                    magnetic_vector_potential_x_unknown_index_neighbor1 = (
                        self._get_magnetic_vector_potential_x_unknown_index(
                            tag_neighbor1))
                    magnetic_vector_potential_x_unknown_index_neighbor2 = (
                        self._get_magnetic_vector_potential_x_unknown_index(
                            tag_neighbor2))
                    magnetic_vector_potential_y_unknown_index_neighbor1 = (
                        self._get_magnetic_vector_potential_y_unknown_index(
                            tag_neighbor1))
                    magnetic_vector_potential_y_unknown_index_neighbor2 = (
                        self._get_magnetic_vector_potential_y_unknown_index(
                            tag_neighbor2))
                    magnetic_flux_density_z_unknown_index_neighbor1 = (
                        self._get_magnetic_flux_density_z_unknown_index(
                            tag_neighbor1))
                    magnetic_flux_density_z_unknown_index_neighbor2 = (
                        self._get_magnetic_flux_density_z_unknown_index(
                            tag_neighbor2))
                    gauge_unknown_index_neighbor1 = (
                        self._get_gauge_unknown_index(tag_neighbor1))
                    gauge_unknown_index_neighbor2 = (
                        self._get_gauge_unknown_index(tag_neighbor2))

                    # Calculate the denominator.
                    denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                   x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                   x * y_neighbor1 - x * y_neighbor2)

                    # Add the coefficients for the equation corresponding to
                    # Gauss's law.
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_x_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_y_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_x_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_y_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_x_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[gauss_law_electric_potential_equation_index,
                      electric_field_y_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                    # Add the coefficients for the equation corresponding to
                    # Faraday's law in the x-direction.
                    A[faraday_law_x_equation_index,
                      electric_potential_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[faraday_law_x_equation_index,
                      electric_potential_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[faraday_law_x_equation_index,
                      electric_potential_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)

                    # Add the coefficients for the equation corresponding to
                    # Faraday's law in the y-direction.
                    A[faraday_law_y_equation_index,
                      electric_potential_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[faraday_law_y_equation_index,
                      electric_potential_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[faraday_law_y_equation_index,
                      electric_potential_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                    # Add the coefficients for the equation corresponding to
                    # Gauss's law for magnetism in the z-direction.
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor1] += (
                          -(y_neighbor2 - y) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor2] += (
                          -(y - y_neighbor1) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index] += (
                          -(y_neighbor1 - y_neighbor2) / denominator)

                    # Add the coefficients for the equation corresponding to
                    # Ampere's law in the x-direction.
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor1] += (
                          -(x - x_neighbor2) / denominator)
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor2] += (
                          -(x_neighbor1 - x) / denominator)
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index] += (
                          -(x_neighbor2 - x_neighbor1) / denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index_neighbor1] += ((y_neighbor2 - y) /
                                                         denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index_neighbor2] += ((y - y_neighbor1) /
                                                         denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index] += ((y_neighbor1 - y_neighbor2) /
                                               denominator)

                    # Add the coefficients for the equation corresponding to
                    # Ampere's law in the y-direction.
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index_neighbor1] += ((x - x_neighbor2) /
                                                         denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index_neighbor2] += ((x_neighbor1 - x) /
                                                         denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index] += ((x_neighbor2 - x_neighbor1) /
                                               denominator)

                    # Add the coefficients for the equation corresponding to
                    # the gauge.
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                # Set the coefficients for the electric fields and magnetic
                # vector potentials for the equations corresponding to
                # Faraday's law.
                A[faraday_law_x_equation_index,
                  electric_field_x_unknown_index] = num_adjacent_triangles
                A[faraday_law_x_equation_index,
                  magnetic_vector_potential_x_unknown_index] = (
                      1j * omega * num_adjacent_triangles)
                A[faraday_law_y_equation_index,
                  electric_field_y_unknown_index] = num_adjacent_triangles
                A[faraday_law_y_equation_index,
                  magnetic_vector_potential_y_unknown_index] = (
                      1j * omega * num_adjacent_triangles)

                # Set the coefficients for the magnetic flux densities for the
                # equation corresponding to Gauss's law for magnetism.
                A[gauss_law_magnetism_z_equation_index,
                  magnetic_flux_density_z_unknown_index] = num_adjacent_triangles

                # Set the coefficients for the electric fields for the
                # equations corresponding to Ampere's law.
                A[ampere_law_x_equation_index,
                  electric_field_x_unknown_index] = (
                      insulator_properties.permeability(self.config.frequency) *
                      insulator_properties.conductivity(self.config.frequency) *
                      num_adjacent_triangles + 1j * omega *
                      insulator_properties.permeability(self.config.frequency) *
                      insulator_properties.permittivity(self.config.frequency) *
                      num_adjacent_triangles)
                A[ampere_law_y_equation_index,
                  electric_field_y_unknown_index] = (
                      insulator_properties.permeability(self.config.frequency) *
                      insulator_properties.conductivity(self.config.frequency) *
                      num_adjacent_triangles + 1j * omega *
                      insulator_properties.permeability(self.config.frequency) *
                      insulator_properties.permittivity(self.config.frequency) *
                      num_adjacent_triangles)

        # Fill in the equations for the nodes within the conductors, revising
        # the equations for the boundary nodes.
        for conductor_tag in conductor_entity_tags:
            conductor_properties = MATERIAL_TO_PROPERTIES[
                self.get_material_for_entity(conductor_tag)]
            _, conductor_triangle_node_tags = self.get_faces(tag=conductor_tag)
            conductor_triangle_neighbors = NeighborLookup(
                conductor_triangle_node_tags)

            # Fill in the equations for the nodes within the conductors.
            conductor_internal_node_tags = self.get_nodes(
                tag=conductor_tag,
                dim=self.dimension(),
                node_type=GmshNodeType.INTERNAL)
            for tag in conductor_internal_node_tags:
                x, y = node_tag_to_coordinates[tag]
                num_adjacent_triangles = (
                    conductor_triangle_neighbors.get_num_adjacent_entities(tag))

                # Equation indices.
                gauss_law_electric_potential_equation_index = (
                    self._get_gauss_law_electric_potential_equation_index(tag))
                faraday_law_x_equation_index = (
                    self._get_faraday_law_x_equation_index(tag))
                faraday_law_y_equation_index = (
                    self._get_faraday_law_y_equation_index(tag))
                gauss_law_magnetism_z_equation_index = (
                    self._get_gauss_law_magnetism_z_equation_index(tag))
                ampere_law_x_equation_index = (
                    self._get_ampere_law_x_equation_index(tag))
                ampere_law_y_equation_index = (
                    self._get_ampere_law_y_equation_index(tag))
                gauge_equation_index = self._get_gauge_equation_index(tag)

                # Unknown indices.
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))
                electric_field_x_unknown_index = (
                    self._get_electric_field_x_unknown_index(tag))
                electric_field_y_unknown_index = (
                    self._get_electric_field_y_unknown_index(tag))
                magnetic_vector_potential_x_unknown_index = (
                    self._get_magnetic_vector_potential_x_unknown_index(tag))
                magnetic_vector_potential_y_unknown_index = (
                    self._get_magnetic_vector_potential_y_unknown_index(tag))
                magnetic_flux_density_z_unknown_index = (
                    self._get_magnetic_flux_density_z_unknown_index(tag))
                gauge_unknown_index = self._get_gauge_unknown_index(tag)

                # Set the voltage boundary conditions.
                A[gauss_law_electric_potential_equation_index,
                  electric_potential_unknown_index] = 1
                b[gauss_law_electric_potential_equation_index] = (
                    self._get_ac_phasor(conductor_tag))

                # Iterate over all adjacent triangles.
                for (tag_neighbor1, tag_neighbor2
                    ) in conductor_triangle_neighbors.get_neighbors(tag):
                    x_neighbor1, y_neighbor1 = node_tag_to_coordinates[
                        tag_neighbor1]
                    x_neighbor2, y_neighbor2 = node_tag_to_coordinates[
                        tag_neighbor2]

                    # Unknown indices for both neighbors.
                    magnetic_vector_potential_x_unknown_index_neighbor1 = (
                        self._get_magnetic_vector_potential_x_unknown_index(
                            tag_neighbor1))
                    magnetic_vector_potential_x_unknown_index_neighbor2 = (
                        self._get_magnetic_vector_potential_x_unknown_index(
                            tag_neighbor2))
                    magnetic_vector_potential_y_unknown_index_neighbor1 = (
                        self._get_magnetic_vector_potential_y_unknown_index(
                            tag_neighbor1))
                    magnetic_vector_potential_y_unknown_index_neighbor2 = (
                        self._get_magnetic_vector_potential_y_unknown_index(
                            tag_neighbor2))
                    magnetic_flux_density_z_unknown_index_neighbor1 = (
                        self._get_magnetic_flux_density_z_unknown_index(
                            tag_neighbor1))
                    magnetic_flux_density_z_unknown_index_neighbor2 = (
                        self._get_magnetic_flux_density_z_unknown_index(
                            tag_neighbor2))
                    gauge_unknown_index_neighbor1 = (
                        self._get_gauge_unknown_index(tag_neighbor1))
                    gauge_unknown_index_neighbor2 = (
                        self._get_gauge_unknown_index(tag_neighbor2))

                    # Calculate the denominator.
                    denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                   x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                   x * y_neighbor1 - x * y_neighbor2)

                    # Add the coefficients for the equation corresponding to
                    # Gauss's law for magnetism in the z-direction.
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor1] += (
                          -(y_neighbor2 - y) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor2] += (
                          -(y - y_neighbor1) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_x_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)
                    A[gauss_law_magnetism_z_equation_index,
                      magnetic_vector_potential_y_unknown_index] += (
                          -(y_neighbor1 - y_neighbor2) / denominator)

                    # Add the coefficients for the equation corresponding to
                    # Ampere's law in the x-direction.
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor1] += (
                          -(x - x_neighbor2) / denominator)
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor2] += (
                          -(x_neighbor1 - x) / denominator)
                    A[ampere_law_x_equation_index,
                      magnetic_flux_density_z_unknown_index] += (
                          -(x_neighbor2 - x_neighbor1) / denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index_neighbor1] += ((y_neighbor2 - y) /
                                                         denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index_neighbor2] += ((y - y_neighbor1) /
                                                         denominator)
                    A[ampere_law_x_equation_index,
                      gauge_unknown_index] += ((y_neighbor1 - y_neighbor2) /
                                               denominator)

                    # Add the coefficients for the equation corresponding to
                    # Ampere's law in the y-direction.
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[ampere_law_y_equation_index,
                      magnetic_flux_density_z_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index_neighbor1] += ((x - x_neighbor2) /
                                                         denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index_neighbor2] += ((x_neighbor1 - x) /
                                                         denominator)
                    A[ampere_law_y_equation_index,
                      gauge_unknown_index] += ((x_neighbor2 - x_neighbor1) /
                                               denominator)

                    # Add the coefficients for the equation corresponding to
                    # the gauge.
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor1] += (
                          (y_neighbor2 - y) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor1] += (
                          (x - x_neighbor2) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index_neighbor2] += (
                          (y - y_neighbor1) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index_neighbor2] += (
                          (x_neighbor1 - x) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_x_unknown_index] += (
                          (y_neighbor1 - y_neighbor2) / denominator)
                    A[gauge_equation_index,
                      magnetic_vector_potential_y_unknown_index] += (
                          (x_neighbor2 - x_neighbor1) / denominator)

                # Set the coefficients for the electric fields and magnetic
                # vector potentials for the equations corresponding to
                # Faraday's law.
                A[faraday_law_x_equation_index,
                  electric_field_x_unknown_index] = 1
                A[faraday_law_x_equation_index,
                  magnetic_vector_potential_x_unknown_index] = 1j * omega
                A[faraday_law_y_equation_index,
                  electric_field_y_unknown_index] = 1
                A[faraday_law_y_equation_index,
                  magnetic_vector_potential_y_unknown_index] = 1j * omega

                # Set the coefficients for the magnetic flux densities for the
                # equation corresponding to Gauss's law for magnetism.
                A[gauss_law_magnetism_z_equation_index,
                  magnetic_flux_density_z_unknown_index] = num_adjacent_triangles

                # Set the coefficients for the electric fields for the
                # equations corresponding to Ampere's law.
                A[ampere_law_x_equation_index,
                  electric_field_x_unknown_index] = (
                      conductor_properties.permeability(self.config.frequency) *
                      conductor_properties.conductivity(self.config.frequency) *
                      num_adjacent_triangles)
                A[ampere_law_y_equation_index,
                  electric_field_y_unknown_index] = (
                      conductor_properties.permeability(self.config.frequency) *
                      conductor_properties.conductivity(self.config.frequency) *
                      num_adjacent_triangles)

            # Set the voltage and magnetic vector potential boundary conditions
            # for the boundary nodes at the capacitor plates.
            conductor_boundary_node_tags = self.get_nodes(
                tag=conductor_tag,
                dim=self.dimension(),
                node_type=GmshNodeType.BOUNDARY)
            for tag in conductor_boundary_node_tags:
                # Equation indices.
                gauss_law_electric_potential_equation_index = (
                    self._get_gauss_law_electric_potential_equation_index(tag))
                gauge_equation_index = self._get_gauge_equation_index(tag)

                # Unknown indices.
                electric_potential_unknown_index = (
                    self._get_electric_potential_unknown_index(tag))
                gauge_unknown_index = self._get_gauge_unknown_index(tag)

                # Set the voltage boundary conditions.
                A[gauss_law_electric_potential_equation_index, :] = 0
                A[gauss_law_electric_potential_equation_index,
                  electric_potential_unknown_index] = 1
                b[gauss_law_electric_potential_equation_index] = (
                    self._get_ac_phasor(conductor_tag))

                # Set the gauge boundary conditions.
                A[gauge_equation_index, :] = 0
                A[gauge_equation_index, gauge_unknown_index] = 1
                b[gauge_equation_index] = 1

        # Solve for the electric potential, the electric field, the magnetic
        # vector potential, and the magnetic flux density.
        x = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        self.electric_potential = x[:self.num_electric_potential_unknowns]
        self.electric_field = np.reshape(
            x[self.num_electric_potential_unknowns:(
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns)], (-1, self.dimension()))
        self.magnetic_vector_potential = np.reshape(
            x[(self.num_electric_potential_unknowns +
               self.num_electric_field_unknowns):(
                   self.num_electric_potential_unknowns +
                   self.num_electric_field_unknowns +
                   self.num_magnetic_vector_potential_unknowns)],
            (-1, self.dimension()))
        self.magnetic_flux_density = np.reshape(
            x[(self.num_electric_potential_unknowns +
               self.num_electric_field_unknowns +
               self.num_magnetic_vector_potential_unknowns):(
                   self.num_electric_potential_unknowns +
                   self.num_electric_field_unknowns +
                   self.num_magnetic_vector_potential_unknowns +
                   self.num_magnetic_flux_density_unknowns)], (-1, 1))

    def _get_electric_potential_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric
        potential.
        """
        return self._get_index_from_tag(tag)

    def _get_electric_field_x_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the x-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns)

    def _get_electric_field_y_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's electric field
        in the y-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) + 1 +
                self.num_electric_potential_unknowns)

    def _get_magnetic_vector_potential_x_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's magnetic
         vector potential in the x-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns)

    def _get_magnetic_vector_potential_y_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's magnetic
         vector potential in the y-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) + 1 +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns)

    def _get_magnetic_flux_density_z_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the node's magnetic flux
        density in the z-direction.
        """
        return (self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_vector_potential_unknowns)

    def _get_gauge_unknown_index(self, tag: int) -> int:
        """Returns the unknown index corresponding to the gauge.
        """
        return (self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_vector_potential_unknowns +
                self.num_magnetic_flux_density_unknowns)

    def _get_gauss_law_electric_potential_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Gauss's law or any
        voltage boundary condition.
        """
        return self._get_index_from_tag(tag)

    def _get_faraday_law_x_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Faraday's law in the
        x-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns)

    def _get_faraday_law_y_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Faraday's law in the
        y-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) + 1 +
                self.num_electric_potential_unknowns)

    def _get_gauss_law_magnetism_z_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Gauss's law for
        magnetism in the z-direction.
        """
        return (self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns)

    def _get_ampere_law_x_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Ampere's law in the
        x-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_flux_density_unknowns)

    def _get_ampere_law_y_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to Ampere's law in the
        y-direction.
        """
        return (self.dimension() * self._get_index_from_tag(tag) + 1 +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_flux_density_unknowns)

    def _get_gauge_equation_index(self, tag: int) -> int:
        """Returns the equation index corresponding to the gauge.
        """
        return (self._get_index_from_tag(tag) +
                self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_flux_density_unknowns +
                self.num_magnetic_vector_potential_unknowns)


class ElectrodynamicSolver3D(ElectrodynamicSolver, ElectromagneticSolver3D):
    """Interface for a 3D electrodynamic solver."""

    pass
