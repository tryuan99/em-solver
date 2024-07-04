"""The capacitor model defines constants for a capacitor model with three
plates.
"""

from proto.capacitor_three_plates_pb2 import CapacitorThreePlatesEntity


class CapacitorThreePlates:
    """Interface for a capacitor with three plates.

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
            ValueError: If the tag does not belong to the structure.
        """
        if tag == CapacitorThreePlatesEntity.GROUND_PLATE or tag == CapacitorThreePlatesEntity.MIDDLE_PLATE:
            return 0
        if tag == CapacitorThreePlatesEntity.VDD_PLATE:
            return self.dc_voltage
        raise ValueError("Invalid capacitor entity tag.")
