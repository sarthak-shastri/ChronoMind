"""
BIMBA Baseline Inference & Handover Pipeline for ReVA VideoQA Challenge.

Architecture:
- Base MLLM: lmms-lab/LLaVA-Video-7B-Qwen2 (revision: 013210b3aff822f1558b166d39c1046dd109520f)
- Adapter: mmiemon/BIMBA-LLaVA-Qwen2-7B (revision: f09f08dcd7401410531d2bb85f5f8c7208821981)
- Selective-scan token compression mechanism based on Mamba.

Official publication:
"BIMBA: Selective-Scan Compression for Long-Range Video Question Answering" (CVPR 2025)
by Md Mohaiminul Islam, Tushar Nagarajan, Huiyu Wang, Gedas Bertasius, Lorenzo Torresani.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Model and checkpoint constants
BASE_MODEL = "lmms-lab/LLaVA-Video-7B-Qwen2"
BASE_REVISION = "013210b3aff822f1558b166d39c1046dd109520f"
ADAPTER_MODEL = "mmiemon/BIMBA-LLaVA-Qwen2-7B"
ADAPTER_REVISION = "f09f08dcd7401410531d2bb85f5f8c7208821981"
DEFAULT_CONV_TEMPLATE = "qwen_1_5"
VALID_LABELS = ("A", "B", "C", "D")

EXPECTED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "peft",
    "decord",
    "av",
]

KNOWN_BLOCKER = (
    "Qwen2 attention received query.dtype=float16 while key/value.dtype=float32 under "
    "bitsandbytes 4-bit NF4 quantization. LoRA adapter layers and non-LoRA trainables "
    "(compressor, mm_projector) default to float32, promoting key/value tensors to float32. "
    "A targeted precision alignment casting these modules to float16 is provided."
)


# ==============================================================================
# 1. Environment & Preflight Utilities
# ==============================================================================

def preflight(repo_path: Optional[Path] = None, video_path: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect environment, GPU availability, and directory validity."""
    result: Dict[str, Any] = {
        "status": "preflight_only",
        "repo_path": str(repo_path) if repo_path else None,
        "repo_exists": repo_path.is_dir() if repo_path else None,
        "video_path": str(video_path) if video_path else None,
        "video_exists": video_path.is_file() if video_path else None,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_model": ADAPTER_MODEL,
        "adapter_revision": ADAPTER_REVISION,
        "packages": {},
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "known_inference_blocker": KNOWN_BLOCKER,
        "prediction": None,
    }

    for package in EXPECTED_PACKAGES:
        try:
            module = __import__(package)
            result["packages"][package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            result["packages"][package] = f"MISSING: {exc}"

    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory
            result["gpu_vram_gb"] = round(total_vram / (1024 ** 3), 2)
    except Exception as exc:
        result["torch_error"] = str(exc)

    return result


# ==============================================================================
# 2. Precision Alignment & Dtype Diagnostics
# ==============================================================================

def diagnose_model_dtypes(model: Any) -> Dict[str, Any]:
    """Audit actual parameter and submodule dtypes across Qwen2 and BIMBA modules.

    Pinpoints modules where float32/float16 divergences cause attention crashes.
    """
    import torch

    report: Dict[str, Any] = {
        "overall_dtype_summary": {},
        "attention_layers_sample": {},
        "lora_modules": {},
        "bimba_modules": {},
        "dtype_mismatch_detected": False,
        "divergent_modules": [],
    }

    dtypes_seen = set()

    for name, param in model.named_parameters():
        dt = str(param.dtype)
        dtypes_seen.add(dt)
        report["overall_dtype_summary"][dt] = report["overall_dtype_summary"].get(dt, 0) + param.numel()

    # Inspect the first Transformer layer's attention projections
    for name, module in model.named_modules():
        # Check first attention block projections
        if "layers.0.self_attn" in name:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(module, proj_name):
                    sub = getattr(module, proj_name)
                    info = {"class": sub.__class__.__name__}
                    if hasattr(sub, "weight") and sub.weight is not None:
                        info["weight_dtype"] = str(sub.weight.dtype)
                        info["weight_device"] = str(sub.weight.device)
                    if hasattr(sub, "lora_A") and "default" in sub.lora_A:
                        info["lora_A_dtype"] = str(sub.lora_A["default"].weight.dtype)
                        info["lora_B_dtype"] = str(sub.lora_B["default"].weight.dtype)
                    report["attention_layers_sample"][f"layers.0.self_attn.{proj_name}"] = info

        # Inspect LoRA layers
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_info = {
                "class": module.__class__.__name__,
                "lora_A_dtype": str(module.lora_A["default"].weight.dtype) if "default" in module.lora_A else None,
                "lora_B_dtype": str(module.lora_B["default"].weight.dtype) if "default" in module.lora_B else None,
            }
            if lora_info["lora_A_dtype"] == "torch.float32":
                report["divergent_modules"].append(f"LoRA in {name} is float32")
                report["dtype_mismatch_detected"] = True
            if len(report["lora_modules"]) < 5:
                report["lora_modules"][name] = lora_info

        # Inspect BIMBA custom multimodal projector & compressor
        if any(key in name for key in ("mm_projector", "compressor")):
            if hasattr(module, "weight") and module.weight is not None:
                m_info = {
                    "dtype": str(module.weight.dtype),
                    "device": str(module.weight.device),
                }
                report["bimba_modules"][name] = m_info
                if module.weight.dtype == torch.float32:
                    report["divergent_modules"].append(f"{name} is float32")
                    report["dtype_mismatch_detected"] = True

    return report


def apply_precision_alignment(model: Any, verbose: bool = True) -> Dict[str, int]:
    """Apply the smallest justified precision fix for Qwen2 attention compatibility.

    NOTE: This is treated as an EXPERIMENTAL precision alignment, not a guaranteed fix.
    It targets only unquantized parameters that default to float32:
    1. Casts LoRA adapter weights (lora_A, lora_B) to float16.
    2. Moves and casts mm_projector and compressor modules to float16 on CUDA.
    3. Leaves 4-bit quantized base linear weights untouched (Linear4bit).
    """
    import torch

    modified_counts = {
        "lora_modules_cast": 0,
        "bimba_modules_cast": 0,
        "bimba_modules_moved_cuda": 0,
    }

    # 1. Cast all LoRA adapter linear weights to float16
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for adapter_key in module.lora_A:
                module.lora_A[adapter_key].to(dtype=torch.float16)
                modified_counts["lora_modules_cast"] += 1
        if hasattr(module, "lora_B"):
            for adapter_key in module.lora_B:
                module.lora_B[adapter_key].to(dtype=torch.float16)
                modified_counts["lora_modules_cast"] += 1

    # 2. Align mm_projector and compressor
    for attr in ("mm_projector", "compressor"):
        # Check in model or model.model
        target = getattr(model, attr, None)
        if target is None and hasattr(model, "model"):
            target = getattr(model.model, attr, None)
        if target is not None:
            target.to(device="cuda", dtype=torch.float16)
            modified_counts["bimba_modules_cast"] += 1
            modified_counts["bimba_modules_moved_cuda"] += 1

    if verbose:
        print(f"[bimba_inference] Precision alignment applied: {modified_counts}")

    return modified_counts


# ==============================================================================
# 3. Video Preprocessing & Sampling
# ==============================================================================

def uniform_indices(frame_count: int, num_frames: int) -> List[int]:
    """Deterministic uniform temporal frame indices."""
    if num_frames <= 0 or frame_count <= 0:
        raise ValueError("frame_count and num_frames must be positive")
    if frame_count == 1:
        return [0] * num_frames
    if num_frames == 1:
        return [0]
    try:
        import numpy as np
        return np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(int).tolist()
    except ImportError:
        return [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]


def resolve_video_file(video_path: Union[str, Path]) -> Path:
    """Resolve video file path, checking current directory and common cache locations."""
    p = Path(video_path)
    if p.is_file():
        return p.resolve()

    filename = p.name
    candidates = [
        Path.cwd() / p,
        Path.cwd() / filename,
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/content/reva_video_cache"),
        Path("/content"),
        Path.cwd(),
    ]
    for root in candidates:
        if root.exists():
            matches = list(root.glob(f"**/{filename}"))
            if matches:
                resolved = matches[0].resolve()
                print(f"[bimba_inference] Resolved video '{video_path}' -> '{resolved}'")
                return resolved

    raise FileNotFoundError(
        f"Video file not found at '{video_path}' and could not be found in cache directories: {candidates}"
    )


def load_video_frames(
    video_path: Union[str, Path],
    num_frames: int = 16,
    fps: int = 1,
    force_sample: bool = True,
) -> Tuple[Any, str, float]:
    """Load video frames adhering to official BIMBA implementation.

    Attempts Decord VideoReader first; falls back gracefully to OpenCV if Decord
    fails or is not installed.
    Returns:
        (frames_numpy, frame_time_string, video_duration_seconds)
    """
    import numpy as np

    resolved_path = resolve_video_file(video_path)
    v_path = str(resolved_path)

    # Primary path: decord (official BIMBA)
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(v_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        avg_fps = float(vr.get_avg_fps() or 24.0)
        duration = total_frames / avg_fps if avg_fps > 0 else 0.0

        if force_sample or total_frames < num_frames:
            frame_indices = uniform_indices(total_frames, num_frames)
        else:
            step = max(1, round(avg_fps / fps))
            frame_indices = [i for i in range(0, total_frames, step)]
            if len(frame_indices) > num_frames:
                frame_indices = uniform_indices(total_frames, num_frames)

        frame_times = [i / avg_fps for i in frame_indices]
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_times])
        frames = vr.get_batch(frame_indices).asnumpy()
        return frames, frame_time_str, duration

    except ImportError:
        pass
    except Exception as exc:
        print(f"[bimba_inference] Decord failed ({exc}); attempting OpenCV fallback...", file=sys.stderr)

    # Fallback path: OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video via OpenCV: {v_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        avg_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        duration = total_frames / avg_fps if avg_fps > 0 and total_frames > 0 else 0.0

        if total_frames <= 0:
            raise RuntimeError(f"Invalid frame count {total_frames} for video: {v_path}")

        indices = uniform_indices(total_frames, num_frames)
        frames_list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed decoding frame {idx} from {v_path}")
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        frames = np.stack(frames_list, axis=0)
        frame_times = [i / avg_fps for i in indices]
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_times])
        return frames, frame_time_str, duration
    except Exception as exc:
        raise RuntimeError(f"Failed to load video {v_path} with both Decord and OpenCV: {exc}") from exc


# ==============================================================================
# 4. Prompt Formatting & Conversation Templates
# ==============================================================================

def format_reva_prompt(
    question: str,
    options: Dict[str, str],
    frame_time_str: Optional[str] = None,
    conv_template: str = DEFAULT_CONV_TEMPLATE,
) -> Tuple[str, str]:
    """Construct the official BIMBA prompt and ChatML conversation string.

    Returns:
        (full_prompt_string, raw_question_block)
    """
    # Format options A, B, C, D
    opts_lines = []
    for label in VALID_LABELS:
        text = options.get(label, "")
        opts_lines.append(f"{label}. {text}")
    formatted_opts = "\n".join(opts_lines)

    # Official BIMBA question structure
    time_prefix = f"The video contains frames sampled at {frame_time_str}.\n" if frame_time_str else ""
    instruction = "Please select the correct option directly from A, B, C, or D."
    question_block = f"{time_prefix}{question}\n{formatted_opts}\n{instruction}"

    # Check if llava conv_templates is available
    try:
        from llava.conversation import conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN
        if conv_template in conv_templates:
            conv = copy.deepcopy(conv_templates[conv_template])
            image_token_prompt = DEFAULT_IMAGE_TOKEN + f"\n{question_block}"
            conv.append_message(conv.roles[0], image_token_prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt(), question_block
    except ImportError:
        pass

    # Exact fallback: Qwen2 / ChatML conversation template
    # System: "You are a helpful assistant."
    # User: "<image>\n{question_block}"
    chatml_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<image>\n{question_block}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return chatml_prompt, question_block


# ==============================================================================
# 5. Model Loading (Tesla T4 4-bit with BIMBA Adapter)
# ==============================================================================

def load_bimba_for_inference(
    repo_path: Optional[Union[str, Path]] = None,
    base_model: str = BASE_MODEL,
    adapter_model: str = ADAPTER_MODEL,
    load_in_4bit: bool = True,
    device: str = "cuda",
    apply_dtype_fix: bool = True,
    offload_dir: str = "./offload",
) -> Tuple[Any, Any, Any, int]:
    """Load BIMBA base model + adapter with 4-bit NF4 quantization for 15GB VRAM.

    Follows the 7 verified steps documented in reports/baseline_handover.md.
    Returns:
        (tokenizer, model, image_processor, context_len)
    """
    if repo_path:
        sys.path.insert(0, str(Path(repo_path).resolve()))

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # 1. Quantization Configuration
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["mm_projector", "compressor"],
        )

    # Attempt to load through official builder if available
    try:
        from llava.model.builder import load_pretrained_model
        print(f"[bimba_inference] Loading via official llava builder from {adapter_model}...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=adapter_model,
            model_base=base_model,
            model_name="llava_qwen_lora",
            load_4bit=load_in_4bit,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            merge_lora=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Official BIMBA loader failed; refusing generic fallback: {exc}"
        ) from exc
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, revision=BASE_REVISION)

        print(f"[bimba_inference] Loading base model: {base_model} (4-bit NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            revision=BASE_REVISION,
            trust_remote_code=True,
        )

        print(f"[bimba_inference] Attaching PEFT adapter: {adapter_model}...")
        os.makedirs(offload_dir, exist_ok=True)
        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            offload_folder=offload_dir,
            revision=ADAPTER_REVISION,
        )

        # Load non_lora_trainables (mm_projector and compressor)
        from huggingface_hub import hf_hub_download
        try:
            trainables_path = hf_hub_download(
                repo_id=adapter_model,
                filename="non_lora_trainables.bin",
                revision=ADAPTER_REVISION,
            )
            print(f"[bimba_inference] Loading non-LoRA trainables from {trainables_path}...")
            state_dict = torch.load(trainables_path, map_location="cpu")
            # Filter non-lora weights
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace("base_model.model.", "").replace("model.", "")
                clean_state_dict[clean_k] = v
            model.load_state_dict(clean_state_dict, strict=False)
        except Exception as t_exc:
            print(f"[bimba_inference] Warning: Could not load non_lora_trainables.bin: {t_exc}", file=sys.stderr)

        image_processor = None
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(base_model, revision=BASE_REVISION)
        except Exception:
            pass

        context_len = getattr(model.config, "max_position_embeddings", 4096)

    # Explicit placement of projector & compressor on CUDA
    for attr in ("mm_projector", "compressor"):
        mod = getattr(model, attr, None)
        if mod is None and hasattr(model, "model"):
            mod = getattr(model.model, attr, None)
        if mod is not None and torch.cuda.is_available():
            mod.to(device="cuda", dtype=torch.float16)

    # Diagnose dtypes before fix
    diag_before = diagnose_model_dtypes(model)
    if diag_before["dtype_mismatch_detected"]:
        print(f"[bimba_inference] DIAGNOSIS: Detected {len(diag_before['divergent_modules'])} divergent float32 modules.")

    # Apply smallest justified precision fix
    if apply_dtype_fix and torch.cuda.is_available():
        apply_precision_alignment(model, verbose=True)

    model.eval()
    return tokenizer, model, image_processor, context_len


# ==============================================================================
# 6. Single-Sample Inference & Parsing
# ==============================================================================

def parse_choice_from_text(raw_text: str) -> Optional[str]:
    """Robust hierarchical parser extracting choice letter A, B, C, or D."""
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # 1. Exact match single letter
    if cleaned in VALID_LABELS:
        return cleaned

    # 2. Leading letter pattern: "A.", "B)", "C:", "(D)", "Option A"
    leading_match = re.search(r"^(?:Option\s*|\()?([A-D])[\b\.\:\)]", cleaned, re.IGNORECASE)
    if leading_match:
        return leading_match.group(1).upper()

    # 3. Explicit statement: "The correct answer is B", "answer: (C)"
    statement_match = re.search(
        r"(?:correct\s*(?:answer|option|choice)\s*is|answer\s*(?:is|:)|option)\s*\(?([A-D])\)?",
        cleaned,
        re.IGNORECASE,
    )
    if statement_match:
        return statement_match.group(1).upper()

    # 4. Fallback search for first standalone choice letter
    standalone_match = re.search(r"\b([A-D])\b", cleaned)
    if standalone_match:
        return standalone_match.group(1).upper()

    return None


def extract_choice_probabilities(
    scores: Sequence[Any],
    tokenizer: Any,
) -> Dict[str, float]:
    """Extract softmax normalized probabilities over choices A, B, C, D from logits."""
    import torch
    import torch.nn.functional as F

    if not scores:
        return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    first_step_logits = scores[0][0]  # Shape: [vocab_size]

    choice_logits = []
    for label in VALID_LABELS:
        # Check both raw letter and space-prefixed letter token IDs
        ids = []
        for candidate in (label, f" {label}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] < len(first_step_logits):
                ids.append(encoded[0])
        if ids:
            max_logit = max(float(first_step_logits[i]) for i in ids)
        else:
            max_logit = float("-inf")
        choice_logits.append(max_logit)

    t_logits = torch.tensor(choice_logits, dtype=torch.float32)
    probs = F.softmax(t_logits, dim=0).tolist()

    return {label: round(float(p), 4) for label, p in zip(VALID_LABELS, probs)}


def predict_single_sample(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    video_path: Union[str, Path],
    question: str,
    options: Dict[str, str],
    sample_id: str = "sample_000001",
    num_frames: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute end-to-end inference on a single ReVA video sample."""
    import torch

    start_time = time.perf_counter()

    # 1. Load video frames
    frames, frame_time_str, duration = load_video_frames(video_path, num_frames=num_frames)

    # 2. Build prompt
    prompt_text, raw_q_block = format_reva_prompt(question, options, frame_time_str=frame_time_str)

    # 3. Preprocess frames through native vision processor
    if image_processor is not None:
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=device, dtype=torch.float16)
        images_input = [video_tensor]
    else:
        # Fallback tensor conversion
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
        images_input = [tensor]

    # 4. Tokenize prompt
    try:
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)
    except ImportError:
        # Direct tokenizer encode fallback
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # 5. Diagnose multimodal embeddings and raw language-model logits
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        base_model = model.get_base_model()

        prepared = base_model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            images_input,
            ["video"],
        )

        prepared_embeds = prepared[4]
        prepared_mask = prepared[2]
        prepared_positions = prepared[1]

        print("\n========== TARGETED MODEL DIAGNOSTICS ==========")

        def tensor_stats(name, tensor):
            t = tensor.detach().float()
            print(
                name,
                "| shape:", tuple(t.shape),
                "| min:", t.min().item(),
                "| max:", t.max().item(),
                "| mean:", t.mean().item(),
                "| std:", t.std().item(),
                "| NaN:", torch.isnan(t).sum().item(),
                "| Inf:", torch.isinf(t).sum().item(),
            )

        tensor_stats("Prepared multimodal embeddings", prepared_embeds)

        # Inspect selected LM-head rows, not the entire weight matrix.
        output_head = base_model.get_output_embeddings()
        weight = output_head.weight

        print("\n========== LM HEAD STRUCTURE ==========")
        print("Output head class:", type(output_head))
        print("Weight class:", type(weight))
        print("Weight dtype:", weight.dtype)
        print("Weight device:", weight.device)
        print("Weight shape:", tuple(weight.shape))
        print("Weight is meta:", weight.is_meta)
        print("Weight requires_grad:", weight.requires_grad)

        print("Has quant_state:", hasattr(weight, "quant_state"))
        if hasattr(weight, "quant_state"):
            qs = weight.quant_state
            print("Quant state:", type(qs))
            print("Quant state is None:", qs is None)
            if qs is not None:
                print("Quantization type:", getattr(qs, "quant_type", None))
                print("Quantization blocksize:", getattr(qs, "blocksize", None))
                print("Quantization dtype:", getattr(qs, "dtype", None))

        embedding = base_model.get_input_embeddings()
        embedding_weight = embedding.weight

        print("Input embedding class:", type(embedding))
        print("Input embedding dtype:", embedding_weight.dtype)
        print("Input embedding shape:", tuple(embedding_weight.shape))
        print(
            "LM head tied to input embedding:",
            weight.data_ptr() == embedding_weight.data_ptr()
            if not weight.is_meta and not embedding_weight.is_meta
            else "not comparable (meta)",
        )

        row_ids_check = torch.tensor(
            [32, 33, 34, 100, 1000, 50000, 100000, 151646],
            device=weight.device,
        )
        col_ids_check = torch.tensor(
            [0, 256, 512, 1024, 1536, 2048, 2560, 3072, 3583],
            device=weight.device,
        )
        sampled_weights = weight.index_select(
            0, row_ids_check
        ).index_select(1, col_ids_check)

        print("Distributed LM-head weight samples:")
        print(sampled_weights.detach().float().cpu().tolist())
        print(
            "Sample absolute maximum:",
            sampled_weights.detach().float().abs().max().item(),
        )

        print("\\n========== FULL WEIGHT CHECK ==========")

        def scan_weight_matrix(name, matrix, chunk_rows=4096):
            total = matrix.numel()
            nonzero = 0
            abs_sum = 0.0
            abs_max = 0.0

            for start in range(0, matrix.shape[0], chunk_rows):
                chunk = matrix[start:start + chunk_rows]
                nonzero += torch.count_nonzero(chunk).item()
                abs_sum += chunk.abs().float().sum().item()
                abs_max = max(abs_max, chunk.abs().max().item())

            print(name)
            print("  Total elements:", total)
            print("  Nonzero elements:", nonzero)
            print("  Zero fraction:", 1.0 - nonzero / total)
            print("  Mean absolute weight:", abs_sum / total)
            print("  Maximum absolute weight:", abs_max)

        # Full matrix scans disabled to reduce T4 overhead.
        # Enable manually only if targeted diagnostics are inconclusive.

        print(
            "tie_word_embeddings:",
            getattr(base_model.config, "tie_word_embeddings", "not set"),
        )

        print(
            "LM head bias:",
            output_head.bias is not None,
        )
        if output_head.bias is not None:
            print(
                "LM head bias abs max:",
                output_head.bias.detach().float().abs().max().item(),
            )

        print("======================================\\n")
        print("======================================\n")

        row_ids = torch.tensor(
            [32, 33, 34, 35, 100, 1000, 50000, 100000, 151646],
            device=weight.device,
        )

        weight_sample = weight.index_select(0, row_ids)[:, :16].detach().float()

        print("LM head weight shape:", tuple(weight.shape))
        print("LM head sample rows:", row_ids.tolist())
        print("LM head sample values:", weight_sample.cpu().tolist())
        print("LM head sample nonzero count:", (weight_sample != 0).sum().item())

        # Run the transformer backbone only, then project the LAST token.
        # This avoids creating logits for all 3,550 positions.
        print("\nRunning targeted last-token forward pass...")

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            backbone_outputs = base_model.model(
                inputs_embeds=prepared_embeds,
                attention_mask=prepared_mask,
                position_ids=prepared_positions,
                use_cache=False,
                return_dict=True,
            )

            last_hidden = backbone_outputs.last_hidden_state[:, -1:, :]

            tensor_stats("Last hidden state", last_hidden)

            last_logits = output_head(last_hidden)

            tensor_stats("Last-token LM logits", last_logits)

            last_logits_1d = last_logits[0, 0].float()
            top_values, top_ids = torch.topk(last_logits_1d, k=10)

            print("Last-token top IDs:", top_ids.cpu().tolist())
            print("Last-token top logits:", top_values.cpu().tolist())
            print(
                "Last-token top decoded:",
                [
                    tokenizer.decode([int(i)], skip_special_tokens=False)
                    for i in top_ids.cpu().tolist()
                ],
            )

            for label in ("A", "B", "C", "D"):
                token_ids = tokenizer.encode(
                    label,
                    add_special_tokens=False,
                )
                print(
                    f"Choice {label}:",
                    [
                        (token_id, float(last_logits_1d[token_id].item()))
                        for token_id in token_ids
                    ],
                )

        del backbone_outputs
        del last_hidden
        del last_logits
        del last_logits_1d
        del prepared_embeds
        del prepared
        del prepared_mask
        del prepared_positions
        del base_model
        del output_head
        del weight_sample

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print("Targeted diagnostics completed.")
        print("Proceeding to actual BIMBA generation...")

    # 6. Execute generation under precision-safe scope
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        generation_outputs = model.generate(
            inputs=input_ids,
            images=images_input,
            modalities=["video"],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=16,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Temporary diagnostics: inspect actual generation behavior.
    print("\n========== GENERATION DIAGNOSTICS ==========")

    sequences = generation_outputs.sequences
    print("Generated token IDs:", sequences[0].detach().cpu().tolist())
    print("Generated sequence shape:", tuple(sequences.shape))
    print("Generation score steps:", len(generation_outputs.scores))

    if generation_outputs.scores:
        first_logits = generation_outputs.scores[0][0].float()
        print("First-step logits shape:", tuple(first_logits.shape))
        print("Logits min:", first_logits.min().item())
        print("Logits max:", first_logits.max().item())
        print("Logits mean:", first_logits.mean().item())
        print("NaN count:", torch.isnan(first_logits).sum().item())
        print("Inf count:", torch.isinf(first_logits).sum().item())

        top_values, top_ids = torch.topk(first_logits, k=10)
        print("Top 10 token IDs:", top_ids.detach().cpu().tolist())
        print("Top 10 logits:", top_values.detach().cpu().tolist())
        print(
            "Top 10 decoded tokens:",
            [repr(tokenizer.decode([int(t)])) for t in top_ids]
        )

        for label in VALID_LABELS:
            ids = tokenizer.encode(label, add_special_tokens=False)
            print(
                f"Choice {label}: token IDs={ids}, "
                f"logits={[first_logits[i].item() for i in ids]}"
            )

    print("============================================\n")

    latency = time.perf_counter() - start_time

    # 6. Decode generated text
    # With inputs_embeds, generation may return a one-token placeholder
    # instead of the complete original prompt. Use generation score count
    # to isolate exactly the newly generated tokens.
    generated_steps = len(generation_outputs.scores)
    if generated_steps > 0:
        gen_tokens = generation_outputs.sequences[0, -generated_steps:]
    else:
        gen_tokens = generation_outputs.sequences[0, 0:0]

    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    print(
        f"[bimba_inference] Sequence shape: {tuple(generation_outputs.sequences.shape)}"
        f" | Prompt length: {input_ids.shape[1]}"
        f" | Generated steps: {generated_steps}"
        f" | Decoded output: {raw_output!r}"
    )

    # 7. Parse choice and extract scores
    parsed_pred = parse_choice_from_text(raw_output)
    scores_dict = extract_choice_probabilities(generation_outputs.scores, tokenizer)

    return {
        "sample_id": sample_id,
        "prediction": parsed_pred,
        "scores": scores_dict,
        "raw_text": raw_output,
        "inference_time_seconds": round(latency, 4),
        "video_duration_seconds": round(duration, 2),
        "num_frames": num_frames,
    }


# ==============================================================================
# 7. Batch & Resumable Inference Runner
# ==============================================================================

def run_batch_inference(
    annotation_path: Union[str, Path],
    video_dir: Union[str, Path],
    output_path: Union[str, Path],
    repo_path: Optional[Union[str, Path]] = None,
    subset: Optional[int] = None,
    num_frames: int = 16,
    model_name: str = "BIMBA-LLaVA-Qwen2-7B",
    model_version: str = "CVPR2025",
    run_id: Optional[str] = None,
    resume: bool = True,
) -> Path:
    """Execute validation inference and export canonical prediction JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_out_file = out_file.parent / f"{out_file.stem}_raw.json"

    # Read annotations
    with open(annotation_path, "r", encoding="utf-8") as f:
        annot_data = json.load(f)
    qa_list = annot_data.get("QA", annot_data) if isinstance(annot_data, dict) else annot_data

    if subset is not None and subset > 0:
        qa_list = qa_list[:subset]

    # Load existing progress if resuming
    existing_preds = {}
    if resume and out_file.is_file():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for rec in saved.get("predictions", []):
                existing_preds[rec["sample_id"]] = rec
            print(f"[bimba_inference] Resuming: found {len(existing_preds)} completed samples.")
        except Exception as exc:
            print(f"[bimba_inference] Could not load resume file ({exc}); starting fresh.", file=sys.stderr)

    # Load model
    tokenizer, model, image_processor, _ = load_bimba_for_inference(repo_path=repo_path)

    canonical_predictions = list(existing_preds.values())
    raw_records = []

    for idx, item in enumerate(qa_list):
        s_id = item.get("qa_id", item.get("sample_id", f"sample_{idx:06d}"))
        if s_id in existing_preds:
            continue

        rel_v_path = item.get("video_path")
        v_full_path = Path(video_dir) / rel_v_path
        question = item.get("question", "")
        options = item.get("options", {})

        print(f"[{idx + 1}/{len(qa_list)}] Inferring sample '{s_id}' on {rel_v_path}...")
        try:
            res = predict_single_sample(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                video_path=v_full_path,
                question=question,
                options=options,
                sample_id=s_id,
                num_frames=num_frames,
            )
            canonical_record = {
                "sample_id": s_id,
                "prediction": res["prediction"],
                "scores": res["scores"],
                "inference_time_seconds": res["inference_time_seconds"],
            }
            canonical_predictions.append(canonical_record)
            raw_records.append({**item, **res})

            # Save periodically
            if len(canonical_predictions) % 10 == 0:
                _save_canonical_json(
                    canonical_predictions, out_file, model_name, model_version, run_id
                )

        except Exception as exc:
            print(f"[bimba_inference] Error on sample '{s_id}': {exc}", file=sys.stderr)
            canonical_predictions.append({
                "sample_id": s_id,
                "prediction": None,
                "scores": None,
                "inference_time_seconds": None,
            })

    # Final save
    _save_canonical_json(canonical_predictions, out_file, model_name, model_version, run_id)
    with open(raw_out_file, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)

    print(f"[bimba_inference] Completed. Saved canonical predictions to: {out_file}")
    return out_file


def _save_canonical_json(
    records: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    model_version: str,
    run_id: Optional[str],
) -> None:
    payload = {
        "schema_version": "1.0",
        "dataset": "ReVA",
        "split": "validation",
        "model": {
            "name": model_name,
            "version": model_version,
            "run_id": run_id or f"run_{int(time.time())}",
        },
        "score_semantics": "probabilities",
        "option_order": ["A", "B", "C", "D"],
        "predictions": records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# 8. Main Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="BIMBA baseline inference and evaluation pipeline")
    parser.add_argument("--repo", help="Path to BIMBA repository")
    parser.add_argument("--video", help="Path to a single ReVA video file")
    parser.add_argument("--question", help="Question text for single-video inference")
    parser.add_argument("--options", help="JSON string or comma-separated options for single-video inference")
    parser.add_argument("--sample-id", default="val_000001", help="Sample identifier for single-video inference")
    parser.add_argument("--annotation", help="Path to ReVA annotation JSON (e.g. data/annotations/val.json)")
    parser.add_argument("--video-dir", help="Directory containing ReVA video files")
    parser.add_argument("--output", default="predictions_bimba.json", help="Path to write canonical predictions JSON")
    parser.add_argument("--subset", type=int, default=None, help="Evaluate on subset of N samples")
    parser.add_argument("--num-frames", type=int, default=16, help="Uniform frames to sample (default: 16)")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight environment checks only")
    parser.add_argument("--diagnose-dtypes", action="store_true", help="Diagnose and print model attention dtypes")
    parser.add_argument("--no-dtype-fix", action="store_true", help="Disable precision alignment fix")

    args = parser.parse_args()

    # Preflight check mode
    if args.preflight_only or (not args.video and not args.annotation):
        report = preflight(
            Path(args.repo) if args.repo else None,
            Path(args.video) if args.video else None,
        )
        print(json.dumps(report, indent=2))
        return

    # Single-sample inference mode
    if args.video:
        try:
            v_path = resolve_video_file(args.video)
        except Exception as exc:
            sys.exit(f"Video file error: {exc}")

        # Parse options
        opts_dict = {}
        if args.options:
            try:
                opts_dict = json.loads(args.options)
            except Exception:
                parts = [p.strip() for p in args.options.split(",") if p.strip()]
                for idx, p in enumerate(parts[:4]):
                    opts_dict[VALID_LABELS[idx]] = p

        question = args.question or "At what point in the video does the road with moving vehicles first appear?"
        if not opts_dict:
            opts_dict = {
                "A": "At 5.0 seconds",
                "B": "At 4.2 seconds",
                "C": "At 2.5 seconds",
                "D": "At 3.8 seconds",
            }

        print(f"[bimba_inference] Initializing single-sample inference on {v_path.name}...")
        tokenizer, model, image_processor, _ = load_bimba_for_inference(
            repo_path=args.repo,
            apply_dtype_fix=not args.no_dtype_fix,
        )

        res = predict_single_sample(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_path=v_path,
            question=question,
            options=opts_dict,
            sample_id=args.sample_id,
            num_frames=args.num_frames,
        )
        print("\n=== Single-Sample Inference Result ===")
        print(json.dumps(res, indent=2))

        # Save canonical predictions JSON for direct reva_eval evaluation
        out_p = Path(args.output)
        canonical_record = {
            "sample_id": res["sample_id"],
            "prediction": res["prediction"],
            "scores": res["scores"],
            "inference_time_seconds": res["inference_time_seconds"],
        }
        _save_canonical_json(
            records=[canonical_record],
            output_path=out_p,
            model_name="BIMBA-LLaVA-Qwen2-7B",
            model_version="CVPR2025",
            run_id="bimba_smoke_test",
        )
        print(f"\n[bimba_inference] Saved canonical prediction to: {out_p.resolve()}")
        return

    # Batch inference mode
    if args.annotation:
        if not args.video_dir:
            sys.exit("Batch inference requires --video-dir pointing to video files.")
        run_batch_inference(
            annotation_path=args.annotation,
            video_dir=args.video_dir,
            output_path=args.output,
            repo_path=args.repo,
            subset=args.subset,
            num_frames=args.num_frames,
        )


if __name__ == "__main__":
    main()
