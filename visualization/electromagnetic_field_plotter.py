"""The electromagnetic field plotter plots the electric potential, the electric
field, the magnetic vector potential, and the magnetic flux density.
"""

from abc import ABC, abstractmethod

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from mesh.gmsh_interface import GmshInterface
from visualization.color_maps import COLOR_MAPS


class ElectromagneticFieldPlotter(GmshInterface, ABC):
    """Interface for an electromagnetic field plotter."""

    def __init__(self, mesh_file: str, csv_file: str) -> None:
        super().__init__()

        # Open the mesh file.
        gmsh.open(mesh_file)

        # Open the CSV file.
        self.electromagnetic_fields = pd.read_csv(csv_file, comment="#")
        (
            self.node_tag_column,
            self.electric_potential_column,
            self.electric_field_x_column,
            self.electric_field_y_column,
            self.electric_field_z_column,
            self.magnetic_vector_potential_x_column,
            self.magnetic_vector_potential_y_column,
            self.magnetic_vector_potential_z_column,
            self.magnetic_flux_density_x_column,
            self.magnetic_flux_density_y_column,
            self.magnetic_flux_density_z_column,
        ) = self.electromagnetic_fields.columns

        # Pandas cannot read complex numbers from a CSV file, so identify all
        # columns with an object data type and convert their values into
        # complex numbers.
        converters = {}
        for column, dtype in self.electromagnetic_fields.dtypes.items():
            if dtype == object:
                converters[column] = np.complex128
        self.electromagnetic_fields = pd.read_csv(csv_file,
                                                  comment="#",
                                                  converters=converters)

        # Get the coordinates of the nodes in the order specified in the CSV file.
        self.node_coordinates = self._get_node_coordinates(
            self.electromagnetic_fields[self.node_tag_column])

    @classmethod
    @abstractmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""

    @abstractmethod
    def plot_electric_potential(self) -> None:
        """Plots the electric potential."""

    @abstractmethod
    def plot_electric_field(self) -> None:
        """Plots the electric field."""

    @abstractmethod
    def plot_magnetic_vector_potential(self) -> None:
        """Plots the magnetic vector potential."""

    @abstractmethod
    def plot_magnetic_flux_density(self) -> None:
        """Plots the magnetic flux density."""

    def _get_node_coordinates(self, node_tag_order: np.ndarray) -> np.ndarray:
        """Gets the node coordinates in the specified order.

        Args:
            node_tag_order: Array of node tags.

        Returns:
            A 2D array consisting of the coordinates of each node.
        """
        node_tags, node_coordinates = self.get_node_coordinates(
            dim=self.dimension())
        node_tag_to_coordinates = dict(zip(node_tags, node_coordinates))

        node_coordinates_ordered = np.zeros((len(node_tag_order), 3))
        for node_tag_index, node_tag in enumerate(node_tag_order):
            node_coordinates_ordered[node_tag_index] = (
                node_tag_to_coordinates[node_tag])
        return node_coordinates_ordered


class ElectromagneticFieldPlotter2D(ElectromagneticFieldPlotter):
    """2D electromagnetic field plotter."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 2

    def plot_electric_potential(self) -> None:
        """Plots the electric potential."""
        X = self.node_coordinates[:, 0]
        Y = self.node_coordinates[:, 1]
        electric_potential = (
            self.electromagnetic_fields[self.electric_potential_column])

        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        surf = ax.plot_trisurf(
            X,
            Y,
            np.real(electric_potential),
            cmap=COLOR_MAPS["parula"],
            antialiased=False,
        )
        ax.set_title(r"Electric potential $V$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$V$")
        ax.view_init(45, -45)
        plt.colorbar(surf)
        plt.show()

    def plot_electric_field(self) -> None:
        """Plots the electric field."""
        X = self.node_coordinates[:, 0]
        Y = self.node_coordinates[:, 1]
        electric_field_x = (
            self.electromagnetic_fields[self.electric_field_x_column])
        electric_field_y = (
            self.electromagnetic_fields[self.electric_field_y_column])

        # Plot the heat map.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        surf = ax.plot_trisurf(
            X,
            Y,
            np.sqrt(np.abs(electric_field_x)**2 + np.abs(electric_field_y)**2),
            cmap=COLOR_MAPS["parula"],
            antialiased=False,
        )
        ax.set_title(r"Electric field $\vec{E}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\vec{E}$")
        ax.view_init(90, -90)
        plt.colorbar(surf)
        plt.show()

        # Plot the vector field.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.quiver(
            X,
            Y,
            np.real(electric_field_x),
            np.real(electric_field_y),
            np.sqrt(np.abs(electric_field_x)**2 + np.abs(electric_field_y)**2),
            angles="xy",
            pivot="middle",
            cmap=COLOR_MAPS["parula"],
        )
        ax.set_title(r"Electric field $\vec{E}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.show()

    def plot_magnetic_vector_potential(self) -> None:
        """Plots the magnetic vector potential."""
        X = self.node_coordinates[:, 0]
        Y = self.node_coordinates[:, 1]
        magnetic_vector_potential_x = (self.electromagnetic_fields[
            self.magnetic_vector_potential_x_column])
        magnetic_vector_potential_y = (self.electromagnetic_fields[
            self.magnetic_vector_potential_y_column])
        magnetic_vector_potential_z = (self.electromagnetic_fields[
            self.magnetic_vector_potential_z_column])

        # Plot the heat map.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        surf = ax.plot_trisurf(
            X,
            Y,
            np.sqrt(
                np.abs(magnetic_vector_potential_x)**2 +
                np.abs(magnetic_vector_potential_y)**2 +
                np.abs(magnetic_vector_potential_z)**2),
            cmap=COLOR_MAPS["parula"],
            antialiased=False,
        )
        ax.set_title(r"Magnetic vector potential $\vec{A}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\vec{A}$")
        ax.view_init(90, -90)
        plt.colorbar(surf)
        plt.show()

        # Plot the vector field.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        ax.quiver(
            X,
            Y,
            0,
            np.real(magnetic_vector_potential_x),
            np.real(magnetic_vector_potential_y),
            np.real(magnetic_vector_potential_z),
            pivot="middle",
            cmap=COLOR_MAPS["parula"],
        )
        ax.set_title(r"Magnetic vector potential $\vec{A}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.view_init(90, -90)
        plt.show()

    def plot_magnetic_flux_density(self) -> None:
        """Plots the magnetic flux density."""
        X = self.node_coordinates[:, 0]
        Y = self.node_coordinates[:, 1]
        magnetic_flux_density_x = (
            self.electromagnetic_fields[self.magnetic_flux_density_x_column])
        magnetic_flux_density_y = (
            self.electromagnetic_fields[self.magnetic_flux_density_y_column])
        magnetic_flux_density_z = (
            self.electromagnetic_fields[self.magnetic_flux_density_z_column])

        # Plot the heat map.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        surf = ax.plot_trisurf(
            X,
            Y,
            np.sqrt(
                np.abs(magnetic_flux_density_x)**2 +
                np.abs(magnetic_flux_density_y)**2 +
                np.abs(magnetic_flux_density_z)**2),
            cmap=COLOR_MAPS["parula"],
            antialiased=False,
        )
        ax.set_title(r"Magnetic flux density $\vec{B}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\vec{B}$")
        ax.view_init(90, -90)
        plt.colorbar(surf)
        plt.show()

        # Plot the vector field.
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "3d"},
        )
        ax.quiver(
            X,
            Y,
            0,
            np.real(magnetic_flux_density_x),
            np.real(magnetic_flux_density_y),
            np.real(magnetic_flux_density_z),
            pivot="middle",
            cmap=COLOR_MAPS["parula"],
        )
        ax.set_title(r"Magnetic flux density $\vec{B}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.view_init(45, -45)
        plt.show()


class ElectromagneticFieldPlotter3D(ElectromagneticFieldPlotter):
    """3D electromagnetic field plotter."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3
