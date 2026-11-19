"""Base adapter interface for ReVA model prediction format conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reva_eval.models import CanonicalPredictions


class BaseAdapter(ABC):
    """Abstract base class for all prediction format adapters."""

    @abstractmethod
    def convert(
        self,
        raw_data: Any,
        model_name: str,
        split: str,
        **kwargs: Any,
    ) -> CanonicalPredictions:
        """Convert raw prediction data into CanonicalPredictions instance."""
        pass

    def convert_file(
        self,
        input_path: Union[str, Path],
        model_name: str,
        split: str,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> CanonicalPredictions:
        """Read raw predictions from a file, convert, and optionally save canonical JSON."""
        in_p = Path(input_path)
        if not in_p.is_file():
            raise FileNotFoundError(f"Input prediction file not found: {in_p}")

        with open(in_p, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        canonical = self.convert(
            raw_data=raw_data,
            model_name=model_name,
            split=split,
            **kwargs,
        )

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(canonical.model_dump(), f, indent=2, ensure_ascii=False)

        return canonical

