"""The material model defines the properties of various materials, including
conductors and insulators.
"""

import numpy as np
from proto.material_pb2 import Material

from model.py.constants import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY


class MaterialProperties:
    """Material properties."""

    def __init__(self, material: Material, relative_permittivity: float,
                 relative_permeability: float, resistivity: float) -> None:
        self._material = material
        self._relative_permittivity = relative_permittivity
        self._relative_permeability = relative_permeability
        self._resistivity = resistivity

    def permittivity(self, frequency: float = 0) -> float:
        """Returns the permittivity at the given frequency in F/m.

        Args:
            frequency: Frequency in Hz.
        """
        return self._relative_permittivity * VACUUM_PERMITTIVITY

    def permeability(self, frequency: float = 0) -> float:
        """Returns the permeability at the given frequency in H/m.

        Args:
            frequency: Frequency in Hz.
        """
        return self._relative_permeability * VACUUM_PERMEABILITY

    def conductivity(self, frequency: float = 0) -> float:
        """Returns the conductivity at the given frequency in S/m."""
        return 1 / self._resistivity

    def resistivity(self, frequency: float = 0) -> float:
        """Returns the resistivity at the given frequency in Ohm*m."""
        return self._resistivity

    @staticmethod
    def is_conductor(material: Material) -> bool:
        """Returns whether the given material is a conductor."""
        return material in [
            Material.CONDUCTOR,
            Material.COPPER,
            Material.GOLD,
        ]

    @staticmethod
    def is_insulator(material: Material) -> bool:
        """Returns whether the given material is an insulator."""
        return material in [
            Material.INSULATOR,
            Material.AIR,
            Material.SILICON_DIOXIDE,
        ]


# Map from the material to its properties.
MATERIAL_TO_PROPERTIES = {
    # Ideal materials.
    Material.INSULATOR:
        MaterialProperties(
            material=Material.INSULATOR,
            relative_permittivity=1,
            relative_permeability=1,
            resistivity=np.inf,
        ),
    Material.CONDUCTOR:
        MaterialProperties(
            material=Material.CONDUCTOR,
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=0,
        ),

    # Insulators.
    Material.AIR:
        MaterialProperties(
            material=Material.AIR,
            relative_permittivity=1.0006,
            relative_permeability=1,
            resistivity=1e13,
        ),
    Material.SILICON_DIOXIDE:
        MaterialProperties(
            material=Material.SILICON_DIOXIDE,
            relative_permittivity=3.9,
            relative_permeability=1,
            resistivity=1e15,
        ),

    # Conductors.
    Material.COPPER:
        MaterialProperties(
            material=Material.COPPER,
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=1.68e-8,
        ),
    Material.GOLD:
        MaterialProperties(
            material=Material.GOLD,
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=2.44e-8,
        ),
}
