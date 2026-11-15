"""Hugging Face encoder model for video-query intent classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class IntentClassifierModel(nn.Module):
	"""Classify a query from the CLS representation of a small text encoder."""

	LABELS = ("VIDEO", "SEGMENT", "CLIP", "FRAME")

	def __init__(self, encoder_name: str = "distilbert-base-uncased") -> None:
		super().__init__()
		self.encoder = AutoModel.from_pretrained(encoder_name)

		config = self.encoder.config
		hidden_size = getattr(config, "hidden_size", getattr(config, "dim", None))
		if hidden_size is None:
			raise ValueError(f"Could not determine hidden size for encoder: {encoder_name}")

		self.classifier = nn.Linear(hidden_size, len(self.LABELS))

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor | None = None,
		**encoder_kwargs: Any,
	) -> torch.Tensor:
		"""Return unnormalized logits ordered as VIDEO, SEGMENT, CLIP, FRAME."""
		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			**encoder_kwargs,
		)
		cls_embedding = outputs.last_hidden_state[:, 0, :]
		return self.classifier(cls_embedding)
