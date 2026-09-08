#!/usr/bin/env python3
"""Dependency-free exact-match/choice evaluation for deterministic benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHOICE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def normalise(text: object, mode: str) -> str:
    value = str(text).strip().lower()
    if mode == "choice":
        match = CHOICE.search(value)
        return match.group(1).lower() if match else value
    if mode == "yes_no":
        if re.search(r"\byes\b", value):
            return "yes"
        if re.search(r"\bno\b", value):
            return "no"
    return " ".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--prediction-key", default="prediction")
    parser.add_argument("--target-key", default="answer")
    parser.add_argument("--mode", choices=("exact", "choice", "yes_no"), default="exact")
    args = parser.parse_args()
    rows = load_jsonl(args.predictions)
    correct = sum(
        normalise(row[args.prediction_key], args.mode) == normalise(row[args.target_key], args.mode)
        for row in rows
    )
    result = {"samples": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

