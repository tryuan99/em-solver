"""The material model defines the properties of various materials, including
conductors and insulators.
"""

import numpy as np
from proto.material_pb2 import Material


class MaterialProperties:
    """Material property."""

    def __init__(self, relative_permittivity: float,
                 resistivity: float) -> None:
        self.relative_permittivity = relative_permittivity
        self.resistivity = resistivity

    def conductivity(self) -> float:
        """Returns the conductivity."""
        return 1 / self.resistivity


# Map from the material to its properties.
MATERIAL_TO_PROPERTIES = {
    # Ideal materials.
    Material.INSULATOR:
        MaterialProperties(
            relative_permittivity=1,
            resistivity=np.inf,
        ),
    Material.CONDUCTOR:
        MaterialProperties(
            relative_permittivity=np.inf,
            resistivity=0,
        ),

    # Insulators.
    Material.AIR:
        MaterialProperties(
            relative_permittivity=1.0006,
            resistivity=1e13,
        ),
    Material.SILICON_DIOXIDE:
        MaterialProperties(
            relative_permittivity=3.9,
            resistivity=1e15,
        ),

    # Conductors.
    Material.COPPER:
        MaterialProperties(
            relative_permittivity=np.inf,
            resistivity=1.68e-8,
        ),
    Material.GOLD:
        MaterialProperties(
            relative_permittivity=np.inf,
            resistivity=2.44e-8,
        ),
}
