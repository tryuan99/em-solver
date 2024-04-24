"""The electrostatic solver solves for the voltage and the electric field only."""

from abc import abstractmethod

from solver.electromagnetic_solver import ElectromagneticSolver


class ElectrostaticSolver(ElectromagneticSolver):
    """Interface for an electrostatic solver."""

    @property
    def num_unknowns(self) -> int:
        """Returns the number of unknowns."""
        return self.num_voltage_unknowns + self.num_electric_field_unknowns


class ElectrostaticSolver2D(ElectrostaticSolver):
    """Interface for a 2D electrostatic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 2


class ElectrostaticSolver3D(ElectrostaticSolver):
    """Interface for a 3D electrostatic solver."""

    @classmethod
    def dimension(cls) -> int:
        """Returns the dimension of the structure."""
        return 3
