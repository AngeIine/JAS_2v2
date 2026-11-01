from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RallySequenceDataset, discover_csv_files, fit_encoders
from models import MultiTaskTemporalGCN


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


OPEN_SYSTEM_SPORTS = {"beachvolleyball"}
CLOSED_SYSTEM_SPORTS = {"badminton", "tennis", "table_tennis"}


def split_files(
    csv_files: Sequence[Path],
    mode: str,
    holdout_sport: str | None,
    transfer_direction: str | None,
):
    if mode == "mixed":
        shuffled = list(csv_files)
        random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.8))
        return shuffled[:split], shuffled[split:]

    if mode == "leave_one_sport_out":
        if not holdout_sport:
            raise ValueError("holdout_sport is required for leave_one_sport_out mode")
        train_files = [p for p in csv_files if p.parent.name != holdout_sport]
        test_files = [p for p in csv_files if p.parent.name == holdout_sport]
        return train_files, test_files

    if mode == "open_closed":
        if transfer_direction == "open_to_closed":
            train_files = [p for p in csv_files if p.parent.name in OPEN_SYSTEM_SPORTS]
            test_files = [p for p in csv_files if p.parent.name in CLOSED_SYSTEM_SPORTS]
        elif transfer_direction == "closed_to_open":
            train_files = [p for p in csv_files if p.parent.name in CLOSED_SYSTEM_SPORTS]
            test_files = [p for p in csv_files if p.parent.name in OPEN_SYSTEM_SPORTS]
        else:
            raise ValueError("transfer_direction must be one of: open_to_closed, closed_to_open")

        if not train_files or not test_files:
            raise ValueError(
                f"Open/closed split produced an empty partition: "
                f"train={len(train_files)}, test={len(test_files)}"
            )
        return train_files, test_files

    raise ValueError(f"Unsupported split mode: {mode}")


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1_scores = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return float(sum(f1_scores) / len(f1_scores))


def run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        score_loss = F.cross_entropy(out["score_logits"], batch["target_score"])
        action_loss = F.cross_entropy(out["action_logits"], batch["target_action"])
        loss = score_loss + action_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_action_classes):
    model.eval()
    score_preds, score_labels = [], []
    action_preds, action_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        score_pred = out["score_logits"].argmax(dim=1).cpu()
        action_pred = out["action_logits"].argmax(dim=1).cpu()

        score_preds.append(score_pred)
        score_labels.append(batch["target_score"].cpu())
        action_preds.append(action_pred)
        action_labels.append(batch["target_action"].cpu())

    score_preds = torch.cat(score_preds)
    score_labels = torch.cat(score_labels)
    action_preds = torch.cat(action_preds)
    action_labels = torch.cat(action_labels)

    score_acc = (score_preds == score_labels).float().mean().item()
    action_acc = (action_preds == action_labels).float().mean().item()

    return {
        "score_acc": score_acc,
        "score_macro_f1": macro_f1(score_preds, score_labels, num_classes=2),
        "action_acc": action_acc,
        "action_macro_f1": macro_f1(action_preds, action_labels, num_classes=num_action_classes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-tier", type=str, default="common", choices=["common", "rich"])
    parser.add_argument(
        "--split-mode",
        type=str,
        default="mixed",
        choices=["mixed", "leave_one_sport_out", "open_closed"],
    )
    parser.add_argument("--holdout-sport", type=str, default=None)
    parser.add_argument(
        "--transfer-direction",
        type=str,
        default=None,
        choices=["open_to_closed", "closed_to_open"],
    )
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_files = discover_csv_files(args.data_root)
    if not csv_files:
        raise ValueError(f"No CSV files found under {args.data_root}")

    train_files, test_files = split_files(
        csv_files,
        args.split_mode,
        args.holdout_sport,
        args.transfer_direction,
    )
    encoders = fit_encoders(csv_files)

    train_dataset = RallySequenceDataset(
        train_files,
        encoders=encoders,
        sequence_len=args.sequence_len,
        feature_tier=args.feature_tier,
    )
    test_dataset = RallySequenceDataset(
        test_files,
        encoders=encoders,
        sequence_len=args.sequence_len,
        feature_tier=args.feature_tier,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskTemporalGCN(
        num_numeric_features=len(train_dataset.numeric_feature_names),
        num_actions=len(encoders.action_to_idx),
        num_positions=len(encoders.position_to_idx),
        num_sports=len(encoders.sport_to_idx),
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"split_mode={args.split_mode}"
        + (f" transfer_direction={args.transfer_direction}" if args.transfer_direction else "")
        + (f" holdout_sport={args.holdout_sport}" if args.holdout_sport else "")
    )
    print(f"train_files={len(train_files)} test_files={len(test_files)}")
    for path in train_files:
        print(f"  train: {path}")
    for path in test_files:
        print(f"  test : {path}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device, len(encoders.action_to_idx))
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"score_acc={metrics['score_acc']:.4f} "
            f"score_f1={metrics['score_macro_f1']:.4f} "
            f"action_acc={metrics['action_acc']:.4f} "
            f"action_f1={metrics['action_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
