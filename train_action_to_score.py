from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FEATURE_COLS = ["target_distance", "pdistance", "pspeed", "bspeed"]
SERVICE_ACTIONS = {"serve", "sv"}
DEFAULT_DATA_FILES = [
    Path("data/beachvolleyball/05_01.csv"),
    Path("data/beachvolleyball/05_02.csv"),
    Path("data/beachvolleyball/06_01.csv"),
    Path("data/beachvolleyball/06_02.csv"),
    Path("data/tennis/zh_sk.csv"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_csv_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    return rows


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        action = row.get("action_name", row.get("name", row.get("action", "")))
        position = str(row.get("position", "")).strip()
        if not action or not position:
            continue

        normalized_row: dict[str, object] = {
            "action": str(action).strip(),
            "position": position,
            "score": safe_int(row.get("score", 0), default=0),
            "start_frame": safe_int(row.get("start_frame", 0), default=0),
            "end_frame": safe_int(row.get("end_frame", row.get("start_frame", 0)), default=0),
        }
        for col in FEATURE_COLS:
            normalized_row[col] = safe_float(row.get(col, 0.0), default=0.0)
        normalized.append(normalized_row)
    return normalized


def build_causal_graph(feature_cols: Sequence[str]) -> np.ndarray:
    dag_adj = np.zeros((len(feature_cols), len(feature_cols)), dtype=np.float32)
    index = {name: idx for idx, name in enumerate(feature_cols)}
    edges = [
        ("target_distance", "pdistance"),
        ("pdistance", "pspeed"),
        ("pspeed", "bspeed"),
    ]
    for src, dst in edges:
        if src in index and dst in index:
            dag_adj[index[src], index[dst]] = 1.0
    return dag_adj


@dataclass
class SequenceRecord:
    rows: list[dict]
    sport_name: str
    source_file: str


def load_sequences(csv_files: Sequence[Path]) -> list[SequenceRecord]:
    sequences: list[SequenceRecord] = []
    for csv_path in csv_files:
        rows = normalize_rows(read_csv_rows(csv_path))
        sport_name = csv_path.parent.name
        current_rows: list[dict] = []

        for row in rows:
            action_name = str(row["action"]).strip().lower()
            if action_name in SERVICE_ACTIONS and current_rows:
                if len(current_rows) >= 2:
                    sequences.append(
                        SequenceRecord(
                            rows=current_rows,
                            sport_name=sport_name,
                            source_file=str(csv_path),
                        )
                    )
                current_rows = [row]
            else:
                current_rows.append(row)

        if len(current_rows) >= 2:
            sequences.append(
                SequenceRecord(
                    rows=current_rows,
                    sport_name=sport_name,
                    source_file=str(csv_path),
                )
            )
    return sequences


def split_sequences(
    sequences: Sequence[SequenceRecord],
    test_ratio: float,
    seed: int,
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:
    if not sequences:
        raise ValueError("No valid action sequences were built from the specified CSV files.")

    shuffled = list(sequences)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    test_size = max(1, int(round(len(shuffled) * test_ratio)))
    if test_size >= len(shuffled):
        test_size = max(1, len(shuffled) - 1)
    train_sequences = shuffled[:-test_size]
    test_sequences = shuffled[-test_size:]
    return train_sequences, test_sequences


def fit_scaler(sequences: Sequence[SequenceRecord]) -> "StandardScalerLite":
    numeric_rows = []
    for seq in sequences:
        for row in seq.rows:
            numeric_rows.append([float(row.get(col, 0.0) or 0.0) for col in FEATURE_COLS])
    return StandardScalerLite(np.asarray(numeric_rows, dtype=np.float32))


class StandardScalerLite:
    def __init__(self, data: np.ndarray) -> None:
        if data.size == 0:
            self.mean_ = np.zeros(len(FEATURE_COLS), dtype=np.float32)
            self.scale_ = np.ones(len(FEATURE_COLS), dtype=np.float32)
        else:
            self.mean_ = data.mean(axis=0).astype(np.float32)
            self.scale_ = data.std(axis=0).astype(np.float32)
            self.scale_[self.scale_ < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.scale_


class ActionScoreDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[SequenceRecord],
        scaler: StandardScalerLite,
        action_to_idx: dict[str, int],
        position_to_idx: dict[str, int],
        sport_to_idx: dict[str, int],
        sequence_len: int,
    ) -> None:
        self.samples = []
        self.scaler = scaler
        self.action_to_idx = action_to_idx
        self.position_to_idx = position_to_idx
        self.sport_to_idx = sport_to_idx
        self.sequence_len = sequence_len

        for seq in sequences:
            sport_idx = sport_to_idx[seq.sport_name]
            for end_idx in range(len(seq.rows)):
                start_idx = max(0, end_idx - sequence_len + 1)
                window = seq.rows[start_idx : end_idx + 1]
                self.samples.append(self._build_sample(window, sport_idx, seq.source_file))

    def _mask_features(self, feature_values: np.ndarray) -> np.ndarray:
        feature_mask = np.ones(len(FEATURE_COLS), dtype=np.float32)
        pd_idx = FEATURE_COLS.index("pdistance")
        ps_idx = FEATURE_COLS.index("pspeed")
        if abs(float(feature_values[pd_idx])) <= 1e-8 and abs(float(feature_values[ps_idx])) <= 1e-8:
            feature_mask[pd_idx] = 0.0
            feature_mask[ps_idx] = 0.0
        return feature_mask

    def _build_sample(self, window: Sequence[dict], sport_idx: int, source_file: str) -> dict:
        seq_len = len(window)
        pad_len = self.sequence_len - seq_len

        numeric_values = []
        feature_masks = []
        prev_actions = []
        positions = []
        sports = []

        start_action_idx = len(self.action_to_idx)
        for row_idx, row in enumerate(window):
            raw_values = np.asarray([float(row.get(col, 0.0) or 0.0) for col in FEATURE_COLS], dtype=np.float32)
            scaled_values = self.scaler.transform(raw_values.reshape(1, -1)).reshape(-1).astype(np.float32)
            feature_mask = self._mask_features(raw_values)
            numeric_values.append(scaled_values * feature_mask)
            feature_masks.append(feature_mask)
            if row_idx == 0:
                prev_actions.append(start_action_idx)
            else:
                prev_actions.append(self.action_to_idx[str(window[row_idx - 1]["action"])])
            positions.append(self.position_to_idx[str(row["position"])])
            sports.append(sport_idx)

        if pad_len > 0:
            zero_numeric = [0.0] * len(FEATURE_COLS)
            zero_mask = [0.0] * len(FEATURE_COLS)
            numeric_values = [zero_numeric] * pad_len + numeric_values
            feature_masks = [zero_mask] * pad_len + feature_masks
            prev_actions = [start_action_idx] * pad_len + prev_actions
            positions = [0] * pad_len + positions
            sports = [sport_idx] * pad_len + sports

        target = window[-1]
        timestep_mask = [0.0] * pad_len + [1.0] * seq_len

        numeric_array = np.asarray(numeric_values, dtype=np.float32)
        feature_mask_array = np.asarray(feature_masks, dtype=np.float32)

        return {
            "numeric": torch.from_numpy(numeric_array),
            "feature_mask": torch.from_numpy(feature_mask_array),
            "prev_action_history": torch.tensor(prev_actions, dtype=torch.long),
            "position_history": torch.tensor(positions, dtype=torch.long),
            "sport_history": torch.tensor(sports, dtype=torch.long),
            "mask": torch.tensor(timestep_mask, dtype=torch.float32),
            "target_action": torch.tensor(self.action_to_idx[str(target["action"])], dtype=torch.long),
            "target_score": torch.tensor(int(target["score"]), dtype=torch.long),
            "source_file": source_file,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class TemporalGraphLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        out = torch.bmm(adj, support)
        out = torch.relu(out)
        out = self.norm(out)
        return self.dropout(out)


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out(attended), weights


class JointActionScoreDAGModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_positions: int,
        num_sports: int,
        num_actions: int,
        dag_adj_matrix: np.ndarray,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dim = max(hidden_dim // 6, 8)
        numeric_dim = hidden_dim - embed_dim * 3
        if numeric_dim <= 0:
            raise ValueError("hidden_dim is too small for numeric and embedding branches")

        self.register_buffer("dag_adj", torch.tensor(dag_adj_matrix, dtype=torch.float32))
        self.feature_mixer = nn.Parameter(torch.eye(num_features, dtype=torch.float32))
        self.prev_action_embed = nn.Embedding(num_actions + 1, embed_dim)
        self.position_embed = nn.Embedding(num_positions, embed_dim)
        self.sport_embed = nn.Embedding(num_sports, embed_dim)
        self.numeric_proj = nn.Linear(num_features, numeric_dim)

        self.gcn1 = TemporalGraphLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gcn2 = TemporalGraphLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.attention = MultiHeadTemporalAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.pool_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def build_temporal_adj(self, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        adj = torch.zeros((batch_size, seq_len, seq_len), device=mask.device)
        for i in range(seq_len):
            adj[:, i, i] = 1.0
            if i > 0:
                adj[:, i, i - 1] = 1.0
                adj[:, i - 1, i] = 1.0
            if i > 1:
                adj[:, i, i - 2] = 1.0
                adj[:, i - 2, i] = 1.0
        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / degree

    def causal_penalty(self) -> torch.Tensor:
        identity = torch.eye(self.dag_adj.size(0), device=self.dag_adj.device)
        allowed = torch.clamp(self.dag_adj + identity, max=1.0)
        forbidden = 1.0 - allowed
        return torch.sum(torch.abs(self.feature_mixer) * forbidden)

    def forward(self, batch: dict) -> dict:
        numeric = batch["numeric"]
        feature_mask = batch["feature_mask"]
        prev_action_hist = batch["prev_action_history"]
        position_hist = batch["position_history"]
        sport_hist = batch["sport_history"]
        time_mask = batch["mask"]

        mixed_numeric = torch.matmul(numeric, self.feature_mixer)
        mixed_numeric = mixed_numeric * torch.clamp(feature_mask, min=0.0, max=1.0)

        x = torch.cat(
            [
                self.numeric_proj(mixed_numeric),
                self.prev_action_embed(prev_action_hist),
                self.position_embed(position_hist),
                self.sport_embed(sport_hist),
            ],
            dim=-1,
        )

        adj = self.build_temporal_adj(time_mask)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        attn_out, attn_weights = self.attention(x, time_mask)
        x = x + attn_out
        x = x + self.ffn(x)

        last_index = time_mask.sum(dim=1).long() - 1
        last_index = torch.clamp(last_index, min=0)
        batch_index = torch.arange(x.size(0), device=x.device)
        last_pooled = x[batch_index, last_index]

        valid_mask = time_mask.unsqueeze(-1)
        mean_pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        max_masked = x.masked_fill(valid_mask == 0, -1e9)
        max_pooled = max_masked.max(dim=1).values
        max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))

        gate_scores = self.pool_gate(x).squeeze(-1)
        gate_scores = gate_scores.masked_fill(time_mask == 0, -1e9)
        gate_weights = torch.softmax(gate_scores, dim=1).unsqueeze(-1)
        attn_pooled = (x * gate_weights).sum(dim=1)

        pooled = self.pool_fuse(torch.cat([last_pooled, mean_pooled, max_pooled, attn_pooled], dim=-1))

        return {
            "score_logits": self.score_head(pooled),
            "action_logits": self.action_head(pooled),
            "attention_weights": attn_weights,
        }


def build_label_maps(sequences: Sequence[SequenceRecord]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    action_names = sorted({str(row["action"]) for seq in sequences for row in seq.rows})
    position_names = sorted({str(row["position"]) for seq in sequences for row in seq.rows})
    sport_names = sorted({seq.sport_name for seq in sequences})
    return (
        {name: idx for idx, name in enumerate(action_names)},
        {name: idx for idx, name in enumerate(position_names)},
        {name: idx for idx, name in enumerate(sport_names)},
    )


def classification_metrics(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    f1 = weighted_f1_score(labels, preds)
    return acc, f1


def regression_style_metrics(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if len(labels) == 0:
        return 0.0, 0.0, 0.0
    errors = preds.astype(np.float32) - labels.astype(np.float32)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    label_mean = float(np.mean(labels.astype(np.float32)))
    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(labels.astype(np.float32) - label_mean)))
    r2 = 0.0 if ss_tot <= 1e-8 else float(1.0 - ss_res / ss_tot)
    return mae, rmse, r2


def weighted_f1_score(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    classes = np.unique(labels)
    total = len(labels)
    weighted_sum = 0.0
    for cls in classes:
        tp = float(np.sum((preds == cls) & (labels == cls)))
        fp = float(np.sum((preds == cls) & (labels != cls)))
        fn = float(np.sum((preds != cls) & (labels == cls)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 0.0 if (precision + recall) <= 1e-8 else 2.0 * precision * recall / (precision + recall)
        support = float(np.sum(labels == cls))
        weighted_sum += f1 * support
    return float(weighted_sum / total)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(
    model: JointActionScoreDAGModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_causal: float,
    score_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        score_loss = F.cross_entropy(out["score_logits"], batch["target_score"])
        action_loss = F.cross_entropy(out["action_logits"], batch["target_action"])
        causal_loss = lambda_causal * model.causal_penalty()
        loss = score_loss_weight * score_loss + action_loss_weight * action_loss + causal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: JointActionScoreDAGModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    score_preds, score_labels = [], []
    action_preds, action_labels = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        score_preds.append(out["score_logits"].argmax(dim=1).cpu().numpy())
        score_labels.append(batch["target_score"].cpu().numpy())
        action_preds.append(out["action_logits"].argmax(dim=1).cpu().numpy())
        action_labels.append(batch["target_action"].cpu().numpy())

    score_preds_np = np.concatenate(score_preds) if score_preds else np.array([], dtype=np.int64)
    score_labels_np = np.concatenate(score_labels) if score_labels else np.array([], dtype=np.int64)
    action_preds_np = np.concatenate(action_preds) if action_preds else np.array([], dtype=np.int64)
    action_labels_np = np.concatenate(action_labels) if action_labels else np.array([], dtype=np.int64)

    score_acc, score_f1 = classification_metrics(score_preds_np, score_labels_np)
    action_acc, action_f1 = classification_metrics(action_preds_np, action_labels_np)
    score_mae, score_rmse, score_r2 = regression_style_metrics(score_preds_np, score_labels_np)
    action_mae, action_rmse, action_r2 = regression_style_metrics(action_preds_np, action_labels_np)

    return {
        "score_acc": score_acc,
        "score_f1": score_f1,
        "score_mae": score_mae,
        "score_rmse": score_rmse,
        "score_r2": score_r2,
        "action_acc": action_acc,
        "action_f1": action_f1,
        "action_mae": action_mae,
        "action_rmse": action_rmse,
        "action_r2": action_r2,
    }


def resolve_data_files(script_dir: Path, user_files: Sequence[str] | None) -> list[Path]:
    file_specs = [Path(p) for p in user_files] if user_files else DEFAULT_DATA_FILES
    resolved = [(script_dir / p).resolve() for p in file_specs]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files:\n" + "\n".join(missing))
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint prediction of action category and scoring outcome with temporal GCN, multi-head attention, and DAG prior."
    )
    parser.add_argument("--data-files", nargs="*", default=None)
    parser.add_argument("--sequence-len", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-causal", type=float, default=1e-5)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    csv_files = resolve_data_files(script_dir, args.data_files)
    all_sequences = load_sequences(csv_files)
    train_sequences, test_sequences = split_sequences(all_sequences, test_ratio=args.test_ratio, seed=args.seed)

    action_to_idx, position_to_idx, sport_to_idx = build_label_maps(all_sequences)
    scaler = fit_scaler(train_sequences)

    train_dataset = ActionScoreDataset(
        train_sequences,
        scaler=scaler,
        action_to_idx=action_to_idx,
        position_to_idx=position_to_idx,
        sport_to_idx=sport_to_idx,
        sequence_len=args.sequence_len,
    )
    test_dataset = ActionScoreDataset(
        test_sequences,
        scaler=scaler,
        action_to_idx=action_to_idx,
        position_to_idx=position_to_idx,
        sport_to_idx=sport_to_idx,
        sequence_len=args.sequence_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointActionScoreDAGModel(
        num_features=len(FEATURE_COLS),
        num_positions=len(position_to_idx),
        num_sports=len(sport_to_idx),
        num_actions=len(action_to_idx),
        dag_adj_matrix=build_causal_graph(FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Using CSV files:")
    for path in csv_files:
        print(f"  {path}")
    print(f"Train sequences: {len(train_sequences)} | Test sequences: {len(test_sequences)}")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Action classes: {len(action_to_idx)} | Position classes: {len(position_to_idx)}")
    print("Feature rule: when pdistance == 0 and pspeed == 0, these two features are masked out for that sample.")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_causal=args.lambda_causal,
            score_loss_weight=args.score_loss_weight,
            action_loss_weight=args.action_loss_weight,
        )
        metrics = evaluate(model=model, loader=test_loader, device=device)
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"score_acc={metrics['score_acc']:.4f} "
            f"score_f1={metrics['score_f1']:.4f} "
            f"action_acc={metrics['action_acc']:.4f} "
            f"action_f1={metrics['action_f1']:.4f} "
            f"score_mae={metrics['score_mae']:.4f} "
            f"score_rmse={metrics['score_rmse']:.4f} "
            f"score_r2={metrics['score_r2']:.4f} "
            f"action_mae={metrics['action_mae']:.4f} "
            f"action_rmse={metrics['action_rmse']:.4f} "
            f"action_r2={metrics['action_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
