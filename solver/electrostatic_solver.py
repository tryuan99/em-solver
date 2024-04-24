"""The electrostatic solver solves for the voltage and the electric field only."""

from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from solver.electromagnetic_solver import ElectromagneticSolver
from visualization.color_maps import COLOR_MAPS


class ElectrostaticSolver(ElectromagneticSolver):
    """Interface for an electrostatic solver."""

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


class ElectrostaticSolver3D(ElectrostaticSolver):
    """Interface for a 3D electrostatic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3
