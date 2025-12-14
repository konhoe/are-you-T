from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "example" / "inference_synthetic.jsonl"

FEEL_WORDS = {
    "feel",
    "feeling",
    "emotion",
    "love",
    "like",
    "heart",
    "emotional",
    "care",
    "compassion",
    "empathy",
    "relationship",
}

THINK_WORDS = {
    "logic",
    "reason",
    "analyze",
    "analysis",
    "objective",
    "data",
    "rational",
    "facts",
    "structure",
    "system",
    "argument",
}

LABEL_MAP: Dict[str, int] = {"F": 0, "T": 1}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}


def score_text(text: str) -> int:
    lowered = str(text).lower()
    f_score = sum(word in lowered for word in FEEL_WORDS)
    t_score = sum(word in lowered for word in THINK_WORDS)
    return 0 if f_score > t_score else 1  # tie -> T


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {path}")

    rows: List[Dict[str, Optional[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", None)
            if text is None or str(text).strip() == "":
                continue

            label = obj.get("label", None)
            rows.append({"text": str(text), "label": label})

    return pd.DataFrame(rows)


def run_eval(df: pd.DataFrame) -> Dict[str, float]:
    gold = [LABEL_MAP[lbl] for lbl in df["label"].tolist()]
    pred = [score_text(txt) for txt in df["text"].tolist()]
    return {
        "acc": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro"),
    }


def main() -> None:
    df = load_jsonl(DATA_PATH)
    if df.empty:
        raise SystemExit(f"No data in {DATA_PATH}")

    has_label = ("label" in df.columns) and df["label"].notna().any()

    if has_label:
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(str).str.strip().str.upper()
        df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

        if df.empty:
            raise SystemExit("No valid labeled rows after filtering (only F/T allowed).")

        metrics = run_eval(df)
        print(f"Baseline Inference: acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} (n={len(df)})")
    else:
        texts = df["text"].tolist()
        preds = [IDX2LABEL[score_text(t)] for t in texts]
        print(f"Baseline Inference (no labels): generated {len(preds)} preds")


if __name__ == "__main__":
    main()
