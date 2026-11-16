"""Synthetic dummy model adapter for interface verification and testing.

IMPORTANT NOTICE:
This module is a clearly labeled dummy adapter designed solely for testing the
ReVAModelInterface lifecycle and automated integration pipelines.
It generates deterministic synthetic predictions without requiring GPU hardware,
deep learning backends, or actual model weights.
IT MUST NEVER BE CONFUSED WITH AN ACTUAL COMPETITION MODEL OR LEADERBOARD RESULT.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from reva_eval.interfaces.model_interface import ReVAModelInterface
from reva_eval.models import GroundTruthSample, PredictionRecord, VALID_LABELS


class DummyBaselineModelAdapter(ReVAModelInterface):
    """Synthetic dummy adapter demonstrating ReVAModelInterface compliance."""

    def __init__(
        self,
        model_name: str = "DummySyntheticBaseline",
        model_version: str = "v0.0.1-synthetic",
        run_id: str = "DUMMY_RUN",
        default_prediction_strategy: str = "deterministic_cycle",
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            run_id=run_id,
            config={
                "synthetic": True,
                "purpose": "interface_verification",
                "strategy": default_prediction_strategy,
            },
        )
        self.strategy = default_prediction_strategy
        self.sample_counter = 0

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Simulate checkpoint loading for interface testing."""
        path = Path(checkpoint_path)
        # Note: Accepts synthetic paths or mock files without requiring PyTorch/GPU
        self.is_loaded = True
        self.config["checkpoint_path"] = str(path)
        self.config["device"] = device

    def predict_sample(
        self,
        sample: GroundTruthSample,
        **kwargs: Any,
    ) -> PredictionRecord:
        """Generate a deterministic synthetic prediction record."""
        t0 = time.perf_counter()
        self.sample_counter += 1

        if self.strategy == "deterministic_cycle":
            label_idx = self.sample_counter % len(VALID_LABELS)
            pred_label = VALID_LABELS[label_idx]
        elif self.strategy == "constant_a":
            pred_label = "A"
        elif self.strategy == "match_ground_truth" and sample.ground_truth:
            pred_label = sample.ground_truth
        else:
            pred_label = "B"

        # Synthetic probability distribution
        scores = {lbl: 0.1 for lbl in VALID_LABELS}
        scores[pred_label] = 0.7

        latency = round(time.perf_counter() - t0 + 0.001, 5)

        return PredictionRecord(
            sample_id=sample.sample_id,
            prediction=pred_label,
            scores=scores,
            inference_time_seconds=latency,
            raw_prediction=pred_label,
        )

