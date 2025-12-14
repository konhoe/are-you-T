from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model.model import MBTIModel, build_model  
from model.encoder import MODEL_NAME, load_tokenizer  
from scripts.processing import (  
    create_loaders,
    load_data,
    set_seed,
    split_dataframe,
)

DATA_PATH = ROOT_DIR / "data/processed/ft_posts.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/ft_classifier"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
CPU_THREADS = max(1, (os.cpu_count() or 1) - 1)
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)
MAX_GRAD_NORM = 1.0

LABEL_MAP = {"F": 0, "T": 1}


def compute_class_weights(train_df) -> torch.Tensor:
    counts = train_df["label"].value_counts()
    total = counts.sum()
    weights = [total / counts.get(lbl, 1) for lbl in LABEL_MAP.keys()]
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(
    model: MBTIModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    loss_fn,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model: MBTIModel, loader: DataLoader, device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    golds: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            gold = labels.detach().cpu().numpy()

            preds.extend(pred.tolist())
            golds.extend(gold.tolist())

    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    return acc, f1, golds, preds


def save_artifacts(model: MBTIModel, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data(DATA_PATH, LABEL_MAP)
    tokenizer = load_tokenizer(MODEL_NAME)

    train_df, val_df, test_df = split_dataframe(df, VAL_RATIO, TEST_RATIO, SEED)
    class_weights = compute_class_weights(train_df).to(device)

    def make_loader(split_df, shuffle):
        labels = [LABEL_MAP[lbl] for lbl in split_df["label"].tolist()]
        enc = tokenizer(
            split_df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        dataset = [
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": torch.tensor(labels[i], dtype=torch.long),
            }
            for i in range(len(labels))
        ]
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    model = build_model().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        val_acc, val_f1, val_gold, val_pred = evaluate(model, val_loader, device)
        pred_dist = np.bincount(val_pred, minlength=2)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f} pred_dist={pred_dist.tolist()}"
        )

    test_acc, test_f1, test_gold, test_pred = evaluate(model, test_loader, device)
    print(f"Test: acc={test_acc:.4f} macro_f1={test_f1:.4f} pred_dist={np.bincount(test_pred, minlength=2).tolist()}")

    save_artifacts(model, tokenizer, OUTPUT_DIR)
    print(f"Saved model and tokenizer to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
