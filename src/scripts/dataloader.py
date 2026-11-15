"""CSV-backed PyTorch dataset and dataloader for intent classification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


LABELS = ("VIDEO", "SEGMENT", "CLIP", "FRAME")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
TARGET_COLUMNS = ("target", "intent", "label", "class", "target_class")


class IntentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
	"""Tokenize CSV queries and return input IDs, mask, and one-hot target."""

	def __init__(
		self,
		csv_path: str | Path,
		tokenizer: Any,
		max_length: int = 128,
		query_column: str = "query",
		target_column: str | None = None,
	) -> None:
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.samples: list[tuple[str, int]] = []

		with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as csv_file:
			reader = csv.DictReader(csv_file)
			fieldnames = reader.fieldnames or []
			if query_column not in fieldnames:
				raise ValueError(f"CSV must contain a {query_column!r} column.")

			if target_column is None:
				target_column = next(
					(column for column in TARGET_COLUMNS if column in fieldnames), None
				)
			if target_column is None or target_column not in fieldnames:
				raise ValueError(
					"CSV must contain a target column named one of: "
					+ ", ".join(TARGET_COLUMNS)
				)

			for row_number, row in enumerate(reader, start=2):
				query = (row.get(query_column) or "").strip()
				label = (row.get(target_column) or "").strip().upper()
				if not query:
					raise ValueError(f"Missing query in CSV row {row_number}.")
				if label not in LABEL_TO_INDEX:
					raise ValueError(
						f"Unknown target {label!r} in CSV row {row_number}; "
						f"expected one of {', '.join(LABELS)}."
					)
				self.samples.append((query, LABEL_TO_INDEX[label]))

		if not self.samples:
			raise ValueError("CSV contains no intent examples.")
		if max_length < 1:
			raise ValueError("max_length must be a positive integer.")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		query, target_index = self.samples[index]
		encoded = self.tokenizer(
			query,
			max_length=self.max_length,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_tensors="pt",
		)
		target = F.one_hot(
			torch.tensor(target_index), num_classes=len(LABELS)
		).to(dtype=torch.float32)
		return (
			encoded["input_ids"].squeeze(0),
			encoded["attention_mask"].squeeze(0),
			target,
		)

