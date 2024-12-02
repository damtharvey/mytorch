# mytorch/nn/modules/__init__.py

from .linear import Linear
from .relu import ReLU
from .flatten import Flatten
from .loss import CrossEntropyLoss

__all__ = ["Linear", "ReLU", "Flatten", "CrossEntropyLoss"]
