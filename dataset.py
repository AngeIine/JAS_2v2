from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


COMMON_NUMERIC_FEATURES = [
    "duration",
]

RICH_OPTIONAL_FEATURES = [
    "num_frames",
    "height",
    "target_distance",
    "pdistance",
    "pspeed",
    "bspeed",
    "grade",
    "next_x1",
    "next_y1",
    "next_x2",
    "next_y2",
    "target_x",
    "target_y",
]


@dataclass
class Encoders:
    action_to_idx: Dict[str, int]
    position_to_idx: Dict[str, int]
    sport_to_idx: Dict[str, int]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _normalize_rows(rows: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for row in rows:
        action = row.get("action_name", row.get("name", row.get("action", "")))
        position = str(row.get("position", "")).strip()
        if not action or not position:
            continue

        start_frame = _safe_float(row.get("start_frame", row.get("startframe")))
        end_frame = _safe_float(
            row.get("end_frame", row.get("endframe", row.get("start_frame", row.get("startframe"))))
        )
        score = _safe_float(row.get("score"))

        normalized_row = {
            "action": str(action).strip(),
            "position": position,
            "score": int(score),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": max(0.0, end_frame - start_frame),
        }

        for col in RICH_OPTIONAL_FEATURES:
            normalized_row[col] = _safe_float(row.get(col, 0.0))

        normalized.append(normalized_row)
    return normalized


def discover_csv_files(data_root: Path) -> List[Path]:
    return sorted(p for p in data_root.rglob("*.csv") if p.is_file())


def fit_encoders(csv_files: Sequence[Path]) -> Encoders:
    actions, positions, sports = set(), set(), set()
    for csv_path in csv_files:
        sport = csv_path.parent.name
        sports.add(sport)
        rows = _normalize_rows(_read_csv_rows(csv_path))
        actions.update(str(row["action"]) for row in rows)
        positions.update(str(row["position"]) for row in rows)

    return Encoders(
        action_to_idx={name: idx for idx, name in enumerate(sorted(actions))},
        position_to_idx={name: idx for idx, name in enumerate(sorted(positions))},
        sport_to_idx={name: idx for idx, name in enumerate(sorted(sports))},
    )


def select_numeric_columns(rows: Sequence[dict], feature_tier: str) -> List[str]:
    numeric_cols = list(COMMON_NUMERIC_FEATURES)
    if feature_tier == "rich" and rows:
        available = set(rows[0].keys())
        numeric_cols.extend([col for col in RICH_OPTIONAL_FEATURES if col in available])
    return numeric_cols


class RallySequenceDataset(Dataset):
    def __init__(
        self,
        csv_files: Sequence[Path],
        encoders: Encoders,
        sequence_len: int = 8,
        feature_tier: str = "common",
    ) -> None:
        self.sequence_len = sequence_len
        self.encoders = encoders
        self.feature_tier = feature_tier
        self.samples = []
        self.numeric_feature_names: List[str] | None = None

        for csv_path in csv_files:
            sport_name = csv_path.parent.name
            sport_idx = encoders.sport_to_idx[sport_name]
            rows = _normalize_rows(_read_csv_rows(csv_path))
            numeric_cols = select_numeric_columns(rows, feature_tier)
            if self.numeric_feature_names is None:
                self.numeric_feature_names = numeric_cols

            records = []
            for row in rows:
                record = dict(row)
                for col in self.numeric_feature_names:
                    record.setdefault(col, 0.0)
                records.append(record)

            for end_idx in range(len(records)):
                start_idx = max(0, end_idx - sequence_len + 1)
                window = records[start_idx : end_idx + 1]
                self.samples.append(
                    self._build_sample(window=window, sport_idx=sport_idx, csv_path=str(csv_path))
                )

        if self.numeric_feature_names is None:
            self.numeric_feature_names = list(COMMON_NUMERIC_FEATURES)

    def _build_sample(self, window: List[dict], sport_idx: int, csv_path: str) -> dict:
        seq_len = len(window)
        pad_len = self.sequence_len - seq_len

        numeric_tensor = []
        action_history = []
        position_history = []
        sport_history = []

        for row in window:
            numeric_values = [float(row.get(col, 0.0) or 0.0) for col in self.numeric_feature_names]
            numeric_tensor.append(numeric_values)
            action_history.append(self.encoders.action_to_idx[str(row["action"])])
            position_history.append(self.encoders.position_to_idx[str(row["position"])])
            sport_history.append(sport_idx)

        if pad_len > 0:
            zero_numeric = [0.0] * len(self.numeric_feature_names)
            numeric_tensor = [zero_numeric] * pad_len + numeric_tensor
            action_history = [0] * pad_len + action_history
            position_history = [0] * pad_len + position_history
            sport_history = [sport_idx] * pad_len + sport_history

        mask = [0] * pad_len + [1] * seq_len
        target = window[-1]

        return {
            "numeric": torch.tensor(numeric_tensor, dtype=torch.float32),
            "action_history": torch.tensor(action_history, dtype=torch.long),
            "position_history": torch.tensor(position_history, dtype=torch.long),
            "sport_history": torch.tensor(sport_history, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "target_action": torch.tensor(self.encoders.action_to_idx[str(target["action"])], dtype=torch.long),
            "target_score": torch.tensor(int(target["score"]), dtype=torch.long),
            "source_file": csv_path,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
