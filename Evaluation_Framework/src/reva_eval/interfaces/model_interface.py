"""Abstract model interface for future model integration in ChronoMind (Role 4).

Defines the standard lifecycle contract for any candidate model (BIMBA, advanced reasoning
models, baselines) to load checkpoints, run inference over ReVA samples, export canonical
predictions, and invoke the evaluation engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from reva_eval.dataset_loader import (
    load_canonical_ground_truth,
    load_raw_annotation_file,
    normalize_annotation,
)
from reva_eval.models import (
    CanonicalGroundTruth,
    CanonicalPredictions,
    EvaluationReport,
    GroundTruthSample,
    ModelMetadata,
    PredictionRecord,
)


class ReVAModelInterface(ABC):
    """Abstract interface defining the model inference and evaluation integration lifecycle."""

    def __init__(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.run_id = run_id
        self.config = config or {}
        self.is_loaded = False

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Load model weights, tokenizer, and vision backbones from checkpoint.

        Subclasses must set self.is_loaded = True upon successful initialization.
        """
        pass

    @abstractmethod
    def predict_sample(
        self,
        sample: GroundTruthSample,
        **kwargs: Any,
    ) -> PredictionRecord:
        """Execute single-sample inference and return a PredictionRecord.

        Args:
            sample: Standard GroundTruthSample containing question, options A-D, video_path, etc.

        Returns:
            PredictionRecord with sample_id, prediction (A, B, C, D), optional scores, and latency.
        """
        pass

    def predict_batch(
        self,
        samples: List[GroundTruthSample],
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        """Execute batch inference over multiple samples.

        Default implementation processes samples sequentially. Subclasses may override
        for batched GPU tensor inference.
        """
        records: List[PredictionRecord] = []
        for s in samples:
            records.append(self.predict_sample(s, **kwargs))
        return records

    def generate_predictions(
        self,
        ground_truth: Union[str, Path, CanonicalGroundTruth],
        output_path: Optional[Union[str, Path]] = None,
        score_semantics: Optional[str] = "probabilities",
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> CanonicalPredictions:
        """Run full split inference and produce canonical predictions.

        Args:
            ground_truth: Path to annotation/ground-truth file or loaded CanonicalGroundTruth.
            output_path: Optional destination path to write canonical predictions JSON.
            score_semantics: Semantics of predicted scores ('probabilities', 'logits', etc.).
            split: Target split name.

        Returns:
            CanonicalPredictions object conforming to canonical contract.
        """
        if isinstance(ground_truth, CanonicalGroundTruth):
            gt = ground_truth
        else:
            gt_p = Path(ground_truth)
            _, _, detected_fmt = load_raw_annotation_file(gt_p)
            if detected_fmt == "canonical":
                gt = load_canonical_ground_truth(gt_p)
            else:
                gt = normalize_annotation(gt_p, split=split or "validation")

        target_split = split or gt.split

        records = self.predict_batch(gt.samples, **kwargs)

        model_meta = ModelMetadata(
            name=self.model_name,
            version=self.model_version,
            run_id=self.run_id,
            metadata=self.config,
        )

        canonical = CanonicalPredictions(
            schema_version="1.0",
            dataset=gt.dataset,
            split=target_split,
            model=model_meta,
            score_semantics=score_semantics,
            option_order=["A", "B", "C", "D"],
            predictions=records,
        )

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(canonical.model_dump(), f, indent=2, ensure_ascii=False)

        return canonical

    def evaluate_model(
        self,
        ground_truth: Union[str, Path, CanonicalGroundTruth],
        output_dir: Optional[Union[str, Path]] = None,
        predictions_output_path: Optional[Union[str, Path]] = None,
        strict_mode: bool = False,
        save_plots: bool = True,
        track_experiment: bool = True,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Convenience method: generate predictions and evaluate them end-to-end."""
        from reva_eval.evaluator import evaluate

        preds = self.generate_predictions(
            ground_truth=ground_truth,
            output_path=predictions_output_path,
            **kwargs,
        )

        return evaluate(
            ground_truth=ground_truth,
            predictions=preds,
            output_dir=output_dir,
            strict_mode=strict_mode,
            save_plots=save_plots,
            track_experiment=track_experiment,
        )

