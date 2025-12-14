from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "processed" / "ft_posts.csv"
TEST_RATIO = 0.2
SEED = 42

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


def score_text(text: str) -> int:
    lowered = text.lower()
    f_score = sum(word in lowered for word in FEEL_WORDS)
    t_score = sum(word in lowered for word in THINK_WORDS)
    return 0 if f_score > t_score else 1  


def run_eval(split_df: pd.DataFrame) -> Dict[str, float]:
    gold = [LABEL_MAP[lbl] for lbl in split_df["label"].tolist()]
    pred = [score_text(txt) for txt in split_df["text"].tolist()]
    return {
        "acc": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro"),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["label", "text"])
    df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=SEED,
    )

    train_metrics = run_eval(train_df)
    test_metrics = run_eval(test_df)

    print(
        f"Baseline Train: acc={train_metrics['acc']:.4f} macro_f1={train_metrics['macro_f1']:.4f} "
        f"(n={len(train_df)})"
    )
    print(
        f"Baseline Test: acc={test_metrics['acc']:.4f} macro_f1={test_metrics['macro_f1']:.4f} "
        f"(n={len(test_df)})"
    )


if __name__ == "__main__":
    main()
