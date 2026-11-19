"""Multimodal multiple-choice video question answering models."""

from .multimodal_qa_model import MultimodalVideoQAModel, answer_index, answer_letter
from .text_encoder import CLIPTextFeatureExtractor

__all__ = ["CLIPTextFeatureExtractor", "MultimodalVideoQAModel", "answer_index", "answer_letter"]
