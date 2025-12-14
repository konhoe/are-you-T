from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple

INPUT_PATH = Path("data/raw/mbti_1.csv")
OUTPUT_PATH = Path("data/processed/ft_posts.csv")
MIN_LENGTH = 25  
LIMIT = None 

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def label_from_mbti(mbti: str) -> Tuple[str, str]:
    stripped = mbti.strip().upper()
    if len(stripped) < 3 or stripped[2] not in {"F", "T"}:
        raise ValueError(f"Cannot derive F/T label from MBTI type '{mbti}'.")
    return stripped[2], stripped


def clean_snippet(snippet: str) -> str:
    text = snippet.strip().strip('"').strip("'")
    text = URL_PATTERN.sub(" ", text)
    text = text.replace("|||", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def iter_clean_rows(
    rows: Iterable[dict],
    min_length: int,
    limit: int | None,
) -> Iterator[Tuple[str, str, str]]:
    emitted = 0
    for row in rows:
        raw_type = row.get("type", "")
        posts = row.get("posts", "")
        try:
            label, normalized_type = label_from_mbti(raw_type)
        except ValueError:
            continue

        for snippet in posts.split("|||"):
            cleaned = clean_snippet(snippet)
            if len(cleaned) < min_length:
                continue
            yield label, cleaned, normalized_type
            emitted += 1
            if limit and emitted >= limit:
                return


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        cleaned_rows = iter_clean_rows(reader, MIN_LENGTH, LIMIT)

        written = 0
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=["label", "text", "source_type"])
            writer.writeheader()
            for label, text, source_type in cleaned_rows:
                writer.writerow(
                    {"label": label, "text": text, "source_type": source_type}
                )
                written += 1

    print(
        f"Wrote {written} cleaned rows to {OUTPUT_PATH} "
        f"(min_length={MIN_LENGTH}, limit={LIMIT})"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
