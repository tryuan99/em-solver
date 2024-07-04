"""The capacitor model defines constants for a capacitor model."""

from abc import abstractmethod
from enum import IntEnum

from proto.capacitor_pb2 import CapacitorEntity


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
        if tag == CapacitorEntity.GROUND_PLATE:
            return 0
        if tag == CapacitorEntity.VDD_PLATE:
            return self.dc_voltage
        raise ValueError("Invalid capacitor entity tag.")

    @abstractmethod
    def calculate_capacitance(self) -> float:
        """Calculates the capacitance."""
