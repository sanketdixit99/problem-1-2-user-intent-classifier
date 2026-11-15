import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from modelling import IntentClassifierModel
from dataloader import IntentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the video intent classifier.")
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = IntentDataset(
        args.train_data_path,
        tokenizer,
        max_length=args.max_length,
    )
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifierModel(encoder_name=args.model_name).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.encoder.eval()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.classifier.train()
        total_loss = 0.0
        for input_ids, attention_mask, targets in tqdm(data, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_indices = targets.argmax(dim=1).to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, target_indices)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        checkpoint_path = args.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(model.classifier.state_dict(), checkpoint_path)
        print(f"Epoch {epoch}/{args.epochs}, loss={total_loss / len(data):.4f}")
        print(f"Saved classifier layer weights to {checkpoint_path}")


if __name__ == "__main__":
    main()

