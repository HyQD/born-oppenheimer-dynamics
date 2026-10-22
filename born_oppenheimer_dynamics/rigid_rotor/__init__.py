"""Time-dependent rigid-rotor dynamics for diatomic molecules."""

from .interface import RigidRotorDiatomic, RotationalState
from .result import RigidRotorResult

__all__ = [
    "RigidRotorDiatomic",
    "RigidRotorResult",
    "RotationalState",
]
