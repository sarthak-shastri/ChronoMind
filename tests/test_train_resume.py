import random
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from advanced_models.qa.multimodal_qa_model import MultimodalVideoQAModel
from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor
from advanced_models.temporal.temporal_encoder import TransformerTemporalEncoder
from advanced_models.vision.feature_extractor import CLIPVisionFeatureExtractor


class MockV(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(()))

    def forward(self, pixel_values):
        return type("Out", (), {"image_embeds": torch.ones(len(pixel_values), 512, device=pixel_values.device)})()


class MockTok:
    def __call__(self, texts, **kw):
        return {"input_ids": torch.ones(len(texts), 3, dtype=torch.long)}


class MockT(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(()))

    def forward(self, input_ids):
        return type("Out", (), {"text_embeds": torch.ones(len(input_ids), 512, device=input_ids.device)})()


def make_model():
    return MultimodalVideoQAModel(
        CLIPVisionFeatureExtractor(MockV(), input_normalization="zero_one"),
        CLIPTextFeatureExtractor(MockT(), MockTok()),
        TransformerTemporalEncoder(512, num_heads=8, num_layers=2, max_seq_len=16, dropout=0.0),
    )


def test_checkpoint_save_and_resume_preserves_state():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "latest.pt"

        # 1. Initialize model and optimizer
        torch.manual_seed(42)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Run 1 dummy training step
        frames = torch.rand(2, 4, 3, 224, 224)
        out = model(frames, ["q1", "q2"], [dict(A="a", B="b", C="c", D="d")] * 2)
        loss = nn.functional.cross_entropy(out["answer_logits"], torch.tensor([0, 1]))
        loss.backward()
        opt.step()
        opt.zero_grad()

        # 2. Save complete checkpoint
        history = [{"epoch": 1, "train_loss": 1.23, "val_accuracy": 0.5}]
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": None,
            "epoch": 1,
            "validation_accuracy": 0.5,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
            "numpy_rng_state": np.random.get_state(),
            "random_state": random.getstate(),
            "model_config": {"temporal_encoder_type": "transformer"},
        }
        torch.save(checkpoint_payload, ckpt_path)
        assert ckpt_path.is_file()

        # 3. Resume into fresh model
        resumed_model = make_model()
        resumed_opt = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)

        ckpt = torch.load(ckpt_path, weights_only=False)
        resumed_model.load_state_dict(ckpt["model_state_dict"])
        resumed_opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        resumed_history = ckpt["history"]

        assert start_epoch == 1
        assert resumed_history == history

        # 4. Verify parameter equivalence
        for p1, p2 in zip(model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(p1, p2)

        # 5. Verify optimizer state equivalence
        for p1_id, p2_id in zip(opt.state.keys(), resumed_opt.state.keys()):
            for k in opt.state[p1_id]:
                if isinstance(opt.state[p1_id][k], torch.Tensor):
                    torch.testing.assert_close(opt.state[p1_id][k], resumed_opt.state[p2_id][k])
