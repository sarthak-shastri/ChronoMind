"""
Unit and integration tests for BIMBA baseline and ReVA evaluation compatibility.

Tests:
1. Syntax and import checks.
2. Frame sampling logic (uniform_indices).
3. Annotation loading and validation split integrity (val.json).
4. Prompt formatting (ChatML/Qwen2 template).
5. Output parser (parse_choice_from_text).
6. Canonical prediction serialization and validation via Pydantic schema.
7. Clear distinction of CPU-verifiable tests vs. tests requiring Tesla T4 GPU.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Evaluation_Framework" / "src"))

from baseline.bimba_inference import (
    BASE_MODEL,
    BASE_REVISION,
    ADAPTER_MODEL,
    ADAPTER_REVISION,
    uniform_indices,
    format_reva_prompt,
    parse_choice_from_text,
    preflight,
    resolve_video_file,
    _save_canonical_json,
)


class TestBimbaBaseline(unittest.TestCase):
    """Test suite for BIMBA baseline utilities."""

    def test_checkpoint_constants(self):
        """Verify official pinned revisions are preserved."""
        self.assertEqual(BASE_MODEL, "lmms-lab/LLaVA-Video-7B-Qwen2")
        self.assertEqual(BASE_REVISION, "013210b3aff822f1558b166d39c1046dd109520f")
        self.assertEqual(ADAPTER_MODEL, "mmiemon/BIMBA-LLaVA-Qwen2-7B")
        self.assertEqual(ADAPTER_REVISION, "f09f08dcd7401410531d2bb85f5f8c7208821981")

    def test_uniform_frame_sampling(self):
        """Verify uniform temporal frame sampling across edge cases."""
        # 16 frames from 100
        idx16 = uniform_indices(100, 16)
        self.assertEqual(len(idx16), 16)
        self.assertEqual(idx16[0], 0)
        self.assertEqual(idx16[-1], 99)
        self.assertTrue(all(idx16[i] <= idx16[i+1] for i in range(len(idx16)-1)))

        # Frame count smaller than num_frames
        idx_small = uniform_indices(5, 8)
        self.assertEqual(len(idx_small), 8)

        # Single frame video
        idx_single = uniform_indices(1, 4)
        self.assertEqual(idx_single, [0, 0, 0, 0])

    def test_prompt_formatting(self):
        """Verify prompt construction with options and timestamps."""
        question = "At what point in the video does the road appear?"
        options = {"A": "5.0s", "B": "4.2s", "C": "2.5s", "D": "3.8s"}
        prompt, block = format_reva_prompt(question, options, frame_time_str="0.0s,1.0s,2.0s")

        self.assertIn("<image>", prompt)
        self.assertIn("At what point in the video does the road appear?", prompt)
        self.assertIn("A. 5.0s", prompt)
        self.assertIn("B. 4.2s", prompt)
        self.assertIn("C. 2.5s", prompt)
        self.assertIn("D. 3.8s", prompt)
        self.assertIn("Please select the correct option directly from A, B, C, or D.", prompt)

    def test_choice_parser(self):
        """Verify robust parsing of raw model text outputs."""
        # Exact letter
        self.assertEqual(parse_choice_from_text("B"), "B")
        self.assertEqual(parse_choice_from_text("  A  "), "A")

        # Leading punctuation / prefix
        self.assertEqual(parse_choice_from_text("B."), "B")
        self.assertEqual(parse_choice_from_text("Option C"), "C")
        self.assertEqual(parse_choice_from_text("(D) 3.8 seconds"), "D")

        # Full sentence responses
        self.assertEqual(parse_choice_from_text("The correct answer is B."), "B")
        self.assertEqual(parse_choice_from_text("Based on the video, option (A) is correct."), "A")
        self.assertEqual(parse_choice_from_text("Answer: C"), "C")

        # None / invalid
        self.assertIsNone(parse_choice_from_text(""))
        self.assertIsNone(parse_choice_from_text("I cannot determine from the video."))

    def test_preflight_output_structure(self):
        """Verify preflight check returns standard fields without raising."""
        report = preflight()
        self.assertEqual(report["status"], "preflight_only")
        self.assertEqual(report["base_model"], BASE_MODEL)
        self.assertEqual(report["adapter_model"], ADAPTER_MODEL)
        self.assertIn("packages", report)
        self.assertIn("known_inference_blocker", report)

    def test_save_canonical_json_single_sample(self):
        """Verify saving a single smoke prediction creates a valid CanonicalPredictions JSON file."""
        import tempfile
        from reva_eval.models import CanonicalPredictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "predictions_smoke.json"
            record = {
                "sample_id": "val_000001",
                "prediction": "B",
                "scores": {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
                "inference_time_seconds": 1.42,
            }
            _save_canonical_json(
                records=[record],
                output_path=out_path,
                model_name="BIMBA-LLaVA-Qwen2-7B",
                model_version="CVPR2025",
                run_id="bimba_smoke_test",
            )
            self.assertTrue(out_path.is_file())
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate via Pydantic model
            model_obj = CanonicalPredictions.model_validate(data)
            self.assertEqual(len(model_obj.predictions), 1)
            self.assertEqual(model_obj.predictions[0].sample_id, "val_000001")
            self.assertEqual(model_obj.predictions[0].prediction, "B")


class TestReVADataAndSchema(unittest.TestCase):
    """Test suite for ReVA annotations and CanonicalPredictions schema validation."""

    def test_val_annotation_integrity(self):
        """Verify the 2,000 validation annotations match official ReVA specifications."""
        val_path = PROJECT_ROOT / "Evaluation_Framework" / "data" / "annotations" / "val.json"
        self.assertTrue(val_path.is_file(), f"val.json not found at {val_path}")

        with open(val_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("QA", data)
        qa = data["QA"]
        self.assertEqual(len(qa), 2000, f"Expected exactly 2000 validation questions, found {len(qa)}")

        # Check balance: exactly 500 of each choice
        counts = {}
        for item in qa:
            ans = item["correct_answer"]
            counts[ans] = counts.get(ans, 0) + 1
            self.assertIn(ans, ("A", "B", "C", "D"))
            self.assertEqual(sorted(item["options"].keys()), ["A", "B", "C", "D"])
            self.assertTrue(item["video_path"].endswith(".mp4"))

        self.assertEqual(counts, {"A": 500, "B": 500, "C": 500, "D": 500})

    def test_canonical_prediction_pydantic_schema(self):
        """Verify that BIMBA prediction format strictly satisfies reva_eval Pydantic models."""
        from reva_eval.models import CanonicalPredictions, PredictionRecord, ModelMetadata

        sample_prediction = {
            "schema_version": "1.0",
            "dataset": "ReVA",
            "split": "validation",
            "model": {
                "name": "BIMBA-LLaVA-Qwen2-7B",
                "version": "CVPR2025",
                "run_id": "bimba_test_run",
            },
            "score_semantics": "probabilities",
            "option_order": ["A", "B", "C", "D"],
            "predictions": [
                {
                    "sample_id": "val_000001",
                    "prediction": "B",
                    "scores": {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
                    "inference_time_seconds": 1.42,
                },
                {
                    "sample_id": "val_000002",
                    "prediction": "A",
                    "scores": {"A": 0.90, "B": 0.03, "C": 0.03, "D": 0.04},
                    "inference_time_seconds": 1.38,
                },
            ],
        }

        # Model validation
        model_obj = CanonicalPredictions.model_validate(sample_prediction)
        self.assertEqual(model_obj.dataset, "ReVA")
        self.assertEqual(len(model_obj.predictions), 2)
        self.assertEqual(model_obj.predictions[0].prediction, "B")
        self.assertEqual(model_obj.predictions[0].scores["B"], 0.85)

    def test_generic_json_adapter_compatibility(self):
        """Verify that GenericJSONAdapter can ingest raw BIMBA output records."""
        from reva_eval.adapters.generic_json import GenericJSONAdapter

        adapter = GenericJSONAdapter()
        raw_bimba_records = [
            {
                "sample_id": "val_000001",
                "prediction": "B",
                "scores": {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
                "inference_time_seconds": 1.42,
            }
        ]

        canonical = adapter.convert(
            raw_data=raw_bimba_records,
            model_name="BIMBA-LLaVA-Qwen2-7B",
            split="validation",
            score_semantics="probabilities",
        )
        self.assertEqual(canonical.model.name, "BIMBA-LLaVA-Qwen2-7B")
        self.assertEqual(canonical.predictions[0].sample_id, "val_000001")
        self.assertEqual(canonical.predictions[0].prediction, "B")


class TestGpuRequirementsBoundary(unittest.TestCase):
    """Document and assert requirements boundary for GPU-dependent execution."""

    def test_gpu_dependencies_boundary(self):
        """Identify components that require live Tesla T4 execution."""
        gpu_requirements = {
            "causal_conv1d_cuda": "Requires CUDA compiler and GPU architecture >= sm_75",
            "selective_scan_cuda": "Requires CUDA compiler and GPU architecture >= sm_75",
            "bitsandbytes_4bit": "Requires CUDA device with >=14 GB VRAM for 7B base model",
            "peft_bimba_adapter": "Requires loaded base model and CUDA device",
            "full_end_to_end_generation": "Requires GPU execution in Colab",
        }
        self.assertTrue(len(gpu_requirements) == 5)
        for comp, reason in gpu_requirements.items():
            self.assertIsInstance(reason, str)


if __name__ == "__main__":
    unittest.main()
