"""Predict an intent label for one query using a saved classifier model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modelling import IntentClassifierModel

DEFAULT_MODEL_PATH = "checkpoints/exp1/epoch_10.pt"
DEFAULT_ENCODER = "distilbert-base-uncased"


def resolve_model_path(model_path: str) -> Path:
	model_path = Path(model_path)
	if model_path.suffix.lower() != ".pt" or not model_path.is_file():
		raise FileNotFoundError(f"PyTorch checkpoint (.pt) not found: {model_path}")
	return model_path


def predict_label(query: str, model_path: str = DEFAULT_MODEL_PATH) -> str:
	"""Return the predicted VIDEO, SEGMENT, CLIP, or FRAME label."""
	if not query.strip():
		raise ValueError("Query must not be empty.")

	checkpoint_path = resolve_model_path(model_path)
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	if isinstance(checkpoint, IntentClassifierModel):
		model = checkpoint
	else:
		model = IntentClassifierModel(encoder_name=DEFAULT_ENCODER)
		model.classifier.load_state_dict(checkpoint)

	model.eval()
	encoder_name = getattr(model.encoder.config, "_name_or_path", DEFAULT_ENCODER)
	tokenizer = AutoTokenizer.from_pretrained(encoder_name)
	encoded = tokenizer(
		query,
		truncation=True,
		max_length=128,
		return_attention_mask=True,
		return_tensors="pt",
	)

	with torch.inference_mode():
		logits = model(
			input_ids=encoded["input_ids"],
			attention_mask=encoded["attention_mask"],
		)
	label_index = int(logits.argmax(dim=-1).item())
	return IntentClassifierModel.LABELS[label_index]


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict a video-query intent label.")
	parser.add_argument("--query", type=str, required=True, help="User query text")
	parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to an existing .pt checkpoint file")
	args = parser.parse_args()
	print(predict_label(args.query, args.model_path))


if __name__ == "__main__":
	main()
