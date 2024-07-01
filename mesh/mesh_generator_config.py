"""The mesh generator configuration specifies the customizable parameters for
mesh generation.
"""


class MeshGeneratorConfig:
    """Mesh generator configuration."""

    def __init__(self, lc_min_factor: float, lc_max_factor: float,
                 distance_min_factor: float,
                 distance_max_factor: float) -> None:
        self.lc_min_factor = lc_min_factor
        self.lc_max_factor = lc_max_factor
        self.distance_min_factor = distance_min_factor
        self.distance_max_factor = distance_max_factor


# Mesh generator fine configuration.
MESH_GENERATOR_FINE_CONFIG = MeshGeneratorConfig(
    lc_min_factor=0.002,
    lc_max_factor=0.1,
    distance_min_factor=0.5,
    distance_max_factor=5,
)

# Mesh generator coarse configuration.
MESH_GENERATOR_COARSE_CONFIG = MeshGeneratorConfig(
    lc_min_factor=0.01,
    lc_max_factor=0.1,
    distance_min_factor=0.5,
    distance_max_factor=5,
)
