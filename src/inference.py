from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model.model import build_model

MODEL_DIR = ROOT_DIR / "outputs" / "ft_classifier"
MAX_LENGTH = 128
BATCH_SIZE = 32

IDX2LABEL = {0: "F", 1: "T"}
LABEL2IDX = {"F": 0, "T": 1}


def load_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    model = build_model()
    state = torch.load(MODEL_DIR / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(texts: Iterable[str]) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device)

    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    preds_idx: List[int] = []
    with torch.no_grad():
        n = enc["input_ids"].shape[0]
        for start in tqdm(range(0, n, BATCH_SIZE), desc="infer", leave=False):
            end = min(start + BATCH_SIZE, n)
            batch = {k: v[start:end] for k, v in enc.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            batch_pred = torch.softmax(logits, dim=1).argmax(dim=1).tolist()
            preds_idx.extend(batch_pred)

    return [IDX2LABEL[i] for i in preds_idx]


def evaluate(texts: Iterable[str], gold_labels: Iterable[str]) -> dict:
    pred_labels = predict(texts)

    gold = list(gold_labels)
    assert len(gold) == len(pred_labels), "gold/pred length mismatch"

    y_true = [LABEL2IDX[g] for g in gold]
    y_pred = [LABEL2IDX[p] for p in pred_labels]

    acc = accuracy_score(y_true, y_pred)
    f1_t = f1_score(y_true, y_pred, pos_label=1)  
    f1_macro = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)  
    report = classification_report(y_true, y_pred, target_names=["F", "T"], digits=4)

    return {
        "acc": acc,
        "f1_t": f1_t,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "pred_labels": pred_labels,
    }


def load_jsonl_texts(path: Path) -> List[str]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return [it["text"] for it in items]


def load_jsonl_texts_and_labels(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            texts.append(it["text"])
            labels.append(it["label"])
    return texts, labels


def write_predictions_jsonl(out_path: Path, texts: List[str], preds: List[str], gold: List[str] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (t, p) in enumerate(zip(texts, preds)):
            rec = {"text": t, "pred": p}
            if gold is not None:
                rec["gold"] = gold[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = ROOT_DIR / "data" / "example" / "inference_synthetic.jsonl"

    first = None
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                first = json.loads(line)
                break

    if first is None:
        raise SystemExit(f"Empty file: {input_path}")

    has_label = isinstance(first, dict) and ("label" in first)

    if has_label:
        texts, gold = load_jsonl_texts_and_labels(input_path)
        metrics = evaluate(texts, gold)
        print(f"Inference: acc={metrics['acc']:.4f} macro_f1={metrics['f1_macro']:.4f}")
    else:
        texts = load_jsonl_texts(input_path)
        preds = predict(texts)
        out_path = ROOT_DIR / "outputs" / "predictions" / "inference_results.jsonl"
        write_predictions_jsonl(out_path, texts, preds, gold=None)
        print(f"Inference (no labels): saved {len(preds)} preds to {out_path}")
