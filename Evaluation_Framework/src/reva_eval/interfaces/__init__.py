"""Model integration interfaces and verification adapters for ReVA VideoQA."""

from reva_eval.interfaces.model_interface import ReVAModelInterface
from reva_eval.interfaces.dummy_adapter import DummyBaselineModelAdapter

__all__ = [
    "ReVAModelInterface",
    "DummyBaselineModelAdapter",
]

