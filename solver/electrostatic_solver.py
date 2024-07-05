"""The electrostatic solver solves for the voltage and the electric field only."""

from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy.sparse
from proto.solver_config_pb2 import SolverConfig

from mesh.gmsh_interface import GmshNodeType
from model.material import MaterialProperties
from solver.electromagnetic_solver import ElectromagneticSolver
from solver.neighbor_lookup import NeighborLookup
from visualization.color_maps import COLOR_MAPS


class ElectrostaticSolver(ElectromagneticSolver):
    """Interface for an electrostatic solver."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__(mesh_file, solver_config)

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return self.num_voltage_unknowns + self.num_electric_field_unknowns

    @abstractmethod
    def plot_voltage(self) -> None:
        """Plots the voltage."""

    @abstractmethod
    def plot_electric_field(self) -> None:
        """Plots the electric field."""

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


class ElectrostaticSolver2D(ElectrostaticSolver):
    """Interface for a 2D electrostatic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 2

    def plot_voltage(self) -> None:
        """Plots the voltage."""
        node_tags, node_coordinates = self.get_node_coordinates(
            dim=self.dimension())
        X = node_coordinates[:, 0]
        Y = node_coordinates[:, 1]
        voltage = self.voltage[self._get_index_from_tag(node_tags)]

        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 8),
            subplot_kw={"projection": "3d"},
        )
        surf = ax.plot_trisurf(
            X,
            Y,
            voltage,
            cmap=COLOR_MAPS["parula"],
            antialiased=False,
        )
        ax.set_title("Voltage")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.view_init(45, -45)
        plt.colorbar(surf)
        plt.show()

    def plot_electric_field(self) -> None:
        """Plots the electric field."""
        node_tags, node_coordinates = self.get_node_coordinates(
            dim=self.dimension())
        X = node_coordinates[:, 0]
        Y = node_coordinates[:, 1]
        electric_field = self.electric_field[self._get_index_from_tag(
            node_tags)]
        electric_field_x = electric_field[:, 0]
        electric_field_y = electric_field[:, 1]

        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.quiver(
            X,
            Y,
            electric_field_x,
            electric_field_y,
            np.linalg.norm(electric_field, axis=1),
            angles="xy",
            pivot="middle",
            cmap=COLOR_MAPS["parula"],
        )
        plt.show()

    def _solve(self) -> None:
        """Implementation for solving for the voltage, the electric field, the
        magnetic vector potential, and the magnetic field.

        In the matrix-vector equation, the first unknowns correspond to the
        nodes' voltages, and the remaining unknowns correspond to the nodes'
        electric fields.
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
        for physical_group_dimension, physical_group_tag in physical_groups:
            material = self.get_material_for_physical_group(
                physical_group_dimension, physical_group_tag)
            entities = self.get_entities_for_physical_group(
                physical_group_dimension, physical_group_tag)
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
                    insulator_triangle_neighbors.get_num_adjacent_entities(tag))

                # Iterate over all adjacent triangles.
                for (tag_neighbor1, tag_neighbor2
                    ) in insulator_triangle_neighbors.get_neighbors(tag):
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

                    # Add the coefficients of the voltages for the electric
                    # field equation in the x-direction.
                    A[electric_field_x_equation_index,
                      voltage_unknown_index_neighbor1] += ((y_neighbor2 - y) /
                                                           denominator)
                    A[electric_field_x_equation_index,
                      voltage_unknown_index_neighbor2] += ((y - y_neighbor1) /
                                                           denominator)
                    A[electric_field_x_equation_index,
                      voltage_unknown_index] += ((y_neighbor1 - y_neighbor2) /
                                                 denominator)

                    # Add the coefficients of the voltages for the electric
                    # field equation in the y-direction.
                    A[electric_field_y_equation_index,
                      voltage_unknown_index_neighbor1] += ((x - x_neighbor2) /
                                                           denominator)
                    A[electric_field_y_equation_index,
                      voltage_unknown_index_neighbor2] += ((x_neighbor1 - x) /
                                                           denominator)
                    A[electric_field_y_equation_index,
                      voltage_unknown_index] += ((x_neighbor2 - x_neighbor1) /
                                                 denominator)

                # Set the coefficient for the electric fields for the
                # electric field equations.
                A[electric_field_x_equation_index,
                  electric_field_x_unknown_index] = -num_adjacent_triangles
                A[electric_field_y_equation_index,
                  electric_field_y_unknown_index] = -num_adjacent_triangles

        # Fill in the voltage and electric field equations for the nodes within
        # the conductors.
        # At steady state, the voltage is constant throughout the insulator,
        # and the electric field is zero throughout, even with a non-zero
        # resistivity.
        for conductor_tag in conductor_entity_tags:
            # Set the voltage boundary conditions and electric fields within
            # the capacitor plates.
            internal_node_tags = self.get_nodes(tag=conductor_tag,
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
                b[voltage_equation_index] = self._get_dc_voltage(conductor_tag)

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
            boundary_node_tags = self.get_nodes(tag=conductor_tag,
                                                dim=self.dimension(),
                                                node_type=GmshNodeType.BOUNDARY)
            for tag in boundary_node_tags:
                voltage_unknown_index = self._get_voltage_unknown_index(tag)

                # Voltage boundary conditions.
                voltage_equation_index = (
                    self._get_poisson_voltage_equation_index(tag))
                A[voltage_equation_index, :] = 0
                A[voltage_equation_index, voltage_unknown_index] = 1
                b[voltage_equation_index] = self._get_dc_voltage(conductor_tag)

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


class ElectrostaticSolver3D(ElectrostaticSolver):
    """Interface for a 3D electrostatic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3
