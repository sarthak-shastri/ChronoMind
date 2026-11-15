import torch
from torch import nn

from advanced_models.qa.text_encoder import CLIPTextFeatureExtractor


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long)}


class _PrecisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, input_ids):
        autocast = torch.is_autocast_enabled(input_ids.device.type)
        dtype = torch.bfloat16 if autocast else torch.float32
        values = torch.full((len(input_ids), 512), 0.3333, dtype=dtype, device=input_ids.device)
        return type("Output", (), {"text_embeds": values})()


def test_text_cache_does_not_reuse_autocast_features_for_fp32_validation():
    extractor = CLIPTextFeatureExtractor(_PrecisionBackbone(), _Tokenizer())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_features = extractor(["same text"])
        amp_again = extractor(["same text"])
    fp32_features = extractor(["same text"])
    fp32_again = extractor(["same text"])

    assert torch.equal(amp_features, amp_again)
    assert torch.equal(fp32_features, fp32_again)
    assert not torch.equal(amp_features, fp32_features)
    assert extractor.feature_cache_stats() == {"entries": 2, "hits": 4, "misses": 2}
