"""Predict one query or evaluate a CSV with a saved classifier model."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataloader import IntentDataset
from modelling import IntentClassifierModel

DEFAULT_MODEL_PATH = "checkpoints/exp1/epoch_10.pt"
DEFAULT_ENCODER = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
OUTPUT_DIR = Path("out")


def resolve_model_path(model_path: str) -> Path:
	model_path = Path(model_path)
	if model_path.suffix.lower() != ".pt" or not model_path.is_file():
		raise FileNotFoundError(f"PyTorch checkpoint (.pt) not found: {model_path}")
	return model_path


def load_model_and_tokenizer(
	model_path: str,
) -> tuple[IntentClassifierModel, AutoTokenizer]:
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
	return model, tokenizer


def predict_label(query: str, model_path: str = DEFAULT_MODEL_PATH) -> str:
	"""Return the predicted VIDEO, SEGMENT, CLIP, or FRAME label."""
	if not query.strip():
		raise ValueError("Query must not be empty.")

	model, tokenizer = load_model_and_tokenizer(model_path)
	encoded = tokenizer(
		query,
		truncation=True,
		max_length=MAX_LENGTH,
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


def prepare_output_dir() -> Path:
	if OUTPUT_DIR.is_symlink():
		raise OSError(f"Output directory must not be a symlink: {OUTPUT_DIR}")
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	for entry in OUTPUT_DIR.iterdir():
		if entry.is_dir() and not entry.is_symlink():
			shutil.rmtree(entry)
		else:
			entry.unlink()
	return OUTPUT_DIR


def write_evaluation_log(records: list[dict[str, str]], output_dir: Path) -> Path:
	log_path = output_dir / "evaluation_log.csv"
	with log_path.open("w", newline="", encoding="utf-8") as log_file:
		writer = csv.DictWriter(log_file, fieldnames=("query", "gt", "predicted"))
		writer.writeheader()
		writer.writerows(records)
	return log_path


def write_confusion_matrix(confusion: list[list[int]], output_path: Path) -> None:
	labels = IntentClassifierModel.LABELS
	cell_size = 72
	left, top = 112, 82
	max_count = max((value for row in confusion for value in row), default=0) or 1
	width = left + cell_size * len(labels) + 24
	height = top + cell_size * len(labels) + 48
	elements = [
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
		'<style>text { font-family: sans-serif; fill: #222; } .label { font-size: 13px; } .count { font-size: 18px; font-weight: bold; }</style>',
		f'<text x="{left + cell_size * 2}" y="24" text-anchor="middle" font-size="18">Confusion Matrix</text>',
		f'<text x="{left + cell_size * 2}" y="54" text-anchor="middle" class="label">Pred</text>',
		f'<text x="22" y="{top + cell_size * 2}" text-anchor="middle" class="label" transform="rotate(-90 22 {top + cell_size * 2})">GT</text>',
	]
	for column, label in enumerate(labels):
		x = left + column * cell_size + cell_size / 2
		elements.append(
			f'<text x="{x}" y="{top - 12}" text-anchor="middle" class="label">{label}</text>'
		)
	for row, label in enumerate(labels):
		y = top + row * cell_size
		elements.append(
			f'<text x="{left - 12}" y="{y + cell_size / 2 + 5}" text-anchor="end" class="label">{label}</text>'
		)
		for column, count in enumerate(confusion[row]):
			x = left + column * cell_size
			intensity = count / max_count
			channel = round(245 - 170 * intensity)
			text_color = "#fff" if intensity > 0.6 else "#222"
			elements.extend(
				(
					f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="rgb({channel},{channel},{255 - round(140 * intensity)})" stroke="#fff"/>',
					f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 6}" text-anchor="middle" class="count" style="fill:{text_color}">{count}</text>',
				)
			)
	elements.append("</svg>")
	output_path.write_text("\n".join(elements), encoding="utf-8")


def evaluate_csv(
	test_csv: str, model_path: str, output_dir: Path
) -> tuple[Path, Path]:
	csv_path = Path(test_csv)
	if csv_path.suffix.lower() != ".csv":
		raise ValueError(f"Test data must be a .csv file: {csv_path}")
	if not csv_path.is_file():
		raise FileNotFoundError(f"Test CSV does not exist: {csv_path}")

	model, tokenizer = load_model_and_tokenizer(model_path)
	dataset = IntentDataset(csv_path, tokenizer, max_length=MAX_LENGTH)
	data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	predictions: list[int] = []
	with torch.inference_mode():
		for input_ids, attention_mask, _ in data:
			logits = model(
				input_ids=input_ids.to(device),
				attention_mask=attention_mask.to(device),
			)
			predictions.extend(logits.argmax(dim=-1).cpu().tolist())

	labels = IntentClassifierModel.LABELS
	confusion = [[0 for _ in labels] for _ in labels]
	records: list[dict[str, str]] = []
	for (query, target_index), prediction_index in zip(dataset.samples, predictions):
		ground_truth = labels[target_index]
		predicted = labels[prediction_index]
		confusion[target_index][prediction_index] += 1
		records.append(
			{"query": query, "gt": ground_truth, "predicted": predicted}
		)

	output_dir = Path("out")
	output_dir.mkdir(parents=True, exist_ok=True)
	log_path = write_evaluation_log(records, output_dir)
	confusion_path = output_dir / "confusion_matrix.svg"
	write_confusion_matrix(confusion, confusion_path)
	return log_path, confusion_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Predict one query or evaluate a CSV of video-query intents."
	)
	parser.add_argument("--query", type=str, help="User query text")
	parser.add_argument(
		"--test_csv",
		type=str,
		help="Path to a labeled .csv file to evaluate",
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default=DEFAULT_MODEL_PATH,
		help="Path to an existing .pt checkpoint file",
	)
	args = parser.parse_args()

	if (args.query is None) == (args.test_csv is None):
		parser.error("pass exactly one of --query or --test_csv")
	output_dir = prepare_output_dir()
	if args.test_csv is not None:
		log_path, confusion_path = evaluate_csv(
			args.test_csv, args.model_path, output_dir
		)
		print(f"Evaluation log saved to {log_path}")
		print(f"Confusion matrix saved to {confusion_path}")
	else:
		predicted = predict_label(args.query, args.model_path)
		log_path = write_evaluation_log(
			[{"query": args.query, "gt": "", "predicted": predicted}],
			output_dir,
		)
		print(predicted)
		print(f"Evaluation log saved to {log_path}")


if __name__ == "__main__":
	main()
