"""The material model defines the properties of various materials, including
conductors and insulators.
"""

import numpy as np
from proto.material_pb2 import Material


class MaterialProperties:
    """Material property."""

    def __init__(self, relative_permittivity: float,
                 relative_permeability: float, resistivity: float) -> None:
        self.relative_permittivity = relative_permittivity
        self.relative_permeability = relative_permeability
        self.resistivity = resistivity

    def conductivity(self) -> float:
        """Returns the conductivity."""
        return 1 / self.resistivity

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
            relative_permittivity=1,
            relative_permeability=1,
            resistivity=np.inf,
        ),
    Material.CONDUCTOR:
        MaterialProperties(
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=0,
        ),

    # Insulators.
    Material.AIR:
        MaterialProperties(
            relative_permittivity=1.0006,
            relative_permeability=1,
            resistivity=1e13,
        ),
    Material.SILICON_DIOXIDE:
        MaterialProperties(
            relative_permittivity=3.9,
            relative_permeability=1,
            resistivity=1e15,
        ),

    # Conductors.
    Material.COPPER:
        MaterialProperties(
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=1.68e-8,
        ),
    Material.GOLD:
        MaterialProperties(
            relative_permittivity=np.inf,
            relative_permeability=1,
            resistivity=2.44e-8,
        ),
}
