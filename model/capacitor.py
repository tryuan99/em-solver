"""The capacitor model defines constants for a capacitor model."""

from enum import IntEnum


class CapacitorEntityTag(IntEnum):
    """Capacitor entity tag enumeration."""
    GROUND_PLATE_TAG = 1
    VDD_PLATE_TAG = 2
    BOUNDING_BOX_TAG = 3


class Capacitor:
    """Interface for a capacitor.

    Attributes:
        dc_voltage: DC voltage applied to the capacitor plates.
    """

    def __init__(self, dc_voltage: float):
        self.dc_voltage = dc_voltage

    def get_dc_voltage(self, tag: int) -> float:
        """Returns the DC voltage for the tag.

        Args:
            tag: Tag of the entity.

        Raises:
            ValueError: If the tag does not belong to the capacitor structure.
        """
        if tag == CapacitorEntityTag.GROUND_PLATE_TAG:
            return 0
        if tag == CapacitorEntityTag.VDD_PLATE_TAG:
            return self.dc_voltage
        raise ValueError("Invalid capacitor entity tag.")
