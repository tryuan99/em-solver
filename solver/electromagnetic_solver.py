"""The electromagnetic solver solves for the voltage, the electric field, the
magnetic vector potential, and the magnetic field."""

from abc import abstractmethod
from typing import Any

import gmsh
import numpy as np

from mesh_generator.gmsh_interface import GmshInterface


class ElectromagneticSolver(GmshInterface):
    """Interface for an electromagnetic solver."""

    def __init__(self, mesh_file: str) -> None:
        super().__init__()

        # Open the mesh file.
        gmsh.open(mesh_file)

        # Initialize the voltage, electric field, and magnetic field vectors.
        num_nodes = len(np.unique(self.get_nodes(dim=self.dimension())))
        self.voltage = np.zeros(num_nodes)
        self.electric_field = np.zeros((num_nodes, self.dimension()))
        self.magnetic_vector_potential = np.zeros((num_nodes, self.dimension()))
        self.magnetic_field = np.zeros((num_nodes, self.dimension()))
        self.solved = False

    @classmethod
    @abstractmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""

    @property
    def num_voltage_unknowns(self) -> int:
        """Returns the number of voltage unknowns."""
        return self.voltage.size

    @property
    def num_electric_field_unknowns(self) -> int:
        """Returns the number of electric field unknowns."""
        return self.electric_field.size

    @property
    def num_magnetic_vector_potential_unknowns(self) -> int:
        """Returns the number of magnetic vector potential unknowns."""
        return self.magnetic_vector_potential.size

    @property
    def num_magnetic_field_unknowns(self) -> int:
        """Returns the number of magnetic field unknowns."""
        return self.magnetic_field.size

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return (self.num_voltage_unknowns + self.num_electric_field_unknowns +
                self.num_magnetic_vector_potential_unknowns +
                self.num_magnetic_field_unknowns)

    def solve(self) -> None:
        """Solves for the voltage, the electric field, the magnetic vector
        potential, and the magnetic field.
        """
        if self.solved:
            return
        self._solve()
        self.solved = True

    @abstractmethod
    def _solve(self) -> None:
        """Implementation for solving for the voltage, the electric field, the
        magnetic vector potential, and the magnetic field.
        """

    @staticmethod
    def _get_index_from_tag(tag: int | Any) -> int | Any:
        """Returns the index corresponding to the node tag.

        Args:
            tag: Node tag.
        """
        return np.int64(tag - 1)

    @staticmethod
    def _get_tag_from_index(index: int | Any) -> int | Any:
        """Returns the node tag corresponding to the index.

        Args:
            index: Node index.
        """
        return np.int64(index + 1)
