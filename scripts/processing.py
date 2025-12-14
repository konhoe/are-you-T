from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MBTIDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_data(path: Path, label_map: Dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["label", "text"])
    df = df[df["label"].isin(label_map.keys())]
    return df.reset_index(drop=True)


def tokenize_texts(tokenizer, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v for k, v in encodings.items()}


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def create_loaders(
    tokenizer,
    df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_map: Dict[str, int],
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df, test_df = split_dataframe(df, val_ratio, test_ratio, seed)

    def to_loader(split_df: pd.DataFrame, shuffle: bool) -> DataLoader:
        labels = [label_map[label] for label in split_df["label"].tolist()]
        encodings = tokenize_texts(tokenizer, split_df["text"].tolist(), max_length)
        dataset = MBTIDataset(encodings, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return to_loader(train_df, True), to_loader(val_df, False), to_loader(test_df, False)
