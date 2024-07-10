"""The electromagnetic solver solves for the electric potential, the electric
field, the magnetic vector potential, and the magnetic flux density.
"""

from abc import ABC, abstractmethod
from typing import Any

import gmsh
import numpy as np
from proto.material_pb2 import Material
from proto.solver_config_pb2 import SolverConfig

from mesh.gmsh_interface import GmshInterface


class ElectromagneticSolver(GmshInterface, ABC):
    """Interface for an electromagnetic solver."""

    def __init__(self, mesh_file: str, solver_config: SolverConfig) -> None:
        super().__init__()
        self.config = solver_config

        # Open the mesh file.
        gmsh.open(mesh_file)

        # Validate the mesh.
        self._validate_mesh()

        # Initialize the electric potential, electric field, magnetic vector
        # potential, and magnetic flux density vectors.
        self.num_nodes = len(np.unique(self.get_nodes(dim=self.dimension())))
        self.electric_potential = np.zeros(self.num_nodes)
        self.electric_field = np.zeros((self.num_nodes, self.dimension()))
        self.magnetic_vector_potential = np.zeros((self.num_nodes, 3))
        self.magnetic_flux_density = np.zeros((self.num_nodes, 3))
        self.solved = False

    @classmethod
    @abstractmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""

    @property
    def num_electric_potential_unknowns(self) -> int:
        """Returns the number of electric potential unknowns."""
        return self.electric_potential.size

    @property
    def num_electric_field_unknowns(self) -> int:
        """Returns the number of electric field unknowns."""
        return self.electric_field.size

    @property
    def num_magnetic_vector_potential_unknowns(self) -> int:
        """Returns the number of magnetic vector potential unknowns."""
        return self.magnetic_vector_potential.size

    @property
    def num_magnetic_flux_density_unknowns(self) -> int:
        """Returns the number of magnetic flux density unknowns."""
        return self.magnetic_flux_density.size

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return (self.num_electric_potential_unknowns +
                self.num_electric_field_unknowns +
                self.num_magnetic_vector_potential_unknowns +
                self.num_magnetic_flux_density_unknowns)

    def get_material_for_physical_group(self, tag: int) -> Material:
        """Returns the material of the physical group.

        Args:
            tag: Tag of the physical group.

        Returns:
            The material of the physical group.
        """
        physical_group_name = gmsh.model.getPhysicalName(dim=self.dimension(),
                                                         tag=tag)
        return Material.Value(physical_group_name)

    def get_material_for_entity(self, tag: int) -> Material:
        """Returns the material of the entity.

        Args:
            tag: Tag of the entity.

        Returns:
            The material of the entity.

        Raises:
            ValueError: If the entity belongs to multiple physical groups.
        """
        physical_group_tags = self.get_physical_groups_for_entity(
            dim=self.dimension(), tag=tag)
        if len(physical_group_tags) > 1:
            raise ValueError(f"Entity {tag} belongs to multiple physical "
                             f"groups.")

        physical_group_tag = physical_group_tags[0]
        return self.get_material_for_physical_group(tag=physical_group_tag)

    def solve(self) -> None:
        """Solves for the electric potential, the electric field, the magnetic
        vector potential, and the magnetic flux density.
        """
        if self.solved:
            return
        self._solve()
        self.solved = True

    def write_solution(self, csv_file: str) -> None:
        """Writes the solution to a CSV file.

        The columns are in the following order:
         - Tag
         - Electric potential
         - Electric field (x, y, z)
         - Magnetic vector potential (x, y, z)
         - Magnetic flux density (x, y, z)
        """
        with open(csv_file, "w") as output:
            output.write(
                f"Tag,Electric potential,"
                f"Electric field x,Electric field y,Electric field z,"
                f"Magnetic vector potential x,Magnetic vector potential y,Magnetic vector potential z,"
                f"Magnetic flux density x,Magnetic flux density y,Magnetic flux density z\n"
            )

            for i in range(self.num_nodes):
                output.write(f"{self._get_tag_from_index(i)},")
                output.write(f"{self.electric_potential[i]},")
                output.write(
                    f"{self.electric_field[i, 0]},"
                    f"{self.electric_field[i, 1]},"
                    f"{self.electric_field[i, 2] if self.dimension() > 2 else 0},"
                )
                output.write(f"{self.magnetic_vector_potential[i, 0]},"
                             f"{self.magnetic_vector_potential[i, 1]},"
                             f"{self.magnetic_vector_potential[i, 2]},")
                output.write(f"{self.magnetic_flux_density[i, 0]},"
                             f"{self.magnetic_flux_density[i, 1]},"
                             f"{self.magnetic_flux_density[i, 2]}")
                output.write("\n")

    def _validate_mesh(self) -> None:
        """Validates the mesh.

        Raises:
            ValueError: If the mesh is invalid and cannot be solved.
        """
        return

    @abstractmethod
    def _solve(self) -> None:
        """Implementation for solving for the electric potential, the electric
        field, the magnetic vector potential, and the magnetic flux density.
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


class ElectromagneticSolver2D(ElectromagneticSolver):
    """Interface for a 2D electromagnetic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 2


class ElectromagneticSolver3D(ElectromagneticSolver):
    """Interface for a 3D electromagnetic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3
