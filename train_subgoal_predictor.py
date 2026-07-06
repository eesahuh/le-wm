"""Train a latent subgoal sequence predictor for PiperX LeWM."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from piper_lerobot_dataset import DEFAULT_DATASET_ROOT
from piper_subgoal_dataset import PiperSubgoalDataset, parse_subgoal_offsets
from subgoal_predictor import (
    LatentSubgoalPredictor,
    build_random_lewm_for_smoke,
    encode_pixels,
    latent_subgoal_loss,
    load_lewm_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PiperX current-history -> future latent subgoal predictor."
    )
    parser.add_argument("--lewm_ckpt", default=None, help="Path to weights_epoch_*.pt or last.ckpt")
    parser.add_argument("--lewm_config", default=None, help="Required only for some Lightning .ckpt files")
    parser.add_argument("--data_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--subgoal_offsets", default="5,10,15,20,30")
    parser.add_argument("--image_key", default="observation.image")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--predictor_depth", type=int, default=2)
    parser.add_argument("--predictor_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine_loss_weight", type=float, default=0.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", default="outputs/subgoal_predictor")
    parser.add_argument("--video_backend", default="auto")
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Run one forward/backward step and exit")
    parser.add_argument(
        "--allow_random_lewm",
        action="store_true",
        help="Use an untrained LeWM-shaped encoder only for local smoke tests.",
    )
    parser.add_argument(
        "--freeze_lewm_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the LeWM encoder while training the subgoal predictor.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    offsets = parse_subgoal_offsets(args.subgoal_offsets)

    dataset = PiperSubgoalDataset(
        root=args.data_root,
        history_size=args.history_size,
        subgoal_offsets=offsets,
        image_key=args.image_key,
        image_size=args.image_size,
        normalize_pixels=True,
        video_backend=args.video_backend,
    )
    train_set, val_set = split_dataset(dataset, args.val_split, args.seed)
    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        device=device,
    )
    val_loader = (
        make_loader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
            device=device,
        )
        if len(val_set) > 0
        else None
    )

    print(f"[subgoal] dataset windows: {len(dataset)}")
    first_sample = dataset[0]
    print(f"[subgoal] history_pixels: {tuple(first_sample['history_pixels'].shape)}")
    print(f"[subgoal] future_pixels: {tuple(first_sample['future_pixels'].shape)}")
    print(f"[subgoal] subgoal_offsets: {offsets}")

    lewm = build_or_load_lewm(args, device)
    if args.freeze_lewm_encoder:
        lewm.eval()
        lewm.requires_grad_(False)
    else:
        lewm.train()

    probe_batch = next(iter(train_loader))
    embed_dim = infer_embed_dim(lewm, probe_batch, device)
    predictor = LatentSubgoalPredictor(
        embed_dim=embed_dim,
        history_size=args.history_size,
        num_subgoals=len(offsets),
        depth=args.predictor_depth,
        heads=args.predictor_heads,
        dropout=args.dropout,
    ).to(device)

    params = list(predictor.parameters())
    if not args.freeze_lewm_encoder:
        params += list(lewm.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.dry_run:
        metrics = train_one_epoch(
            lewm,
            predictor,
            train_loader,
            optimizer,
            device,
            freeze_lewm_encoder=args.freeze_lewm_encoder,
            cosine_loss_weight=args.cosine_loss_weight,
            limit_batches=1,
        )
        print(f"[subgoal] dry_run train_loss={metrics['loss']:.6f}")
        print("[subgoal] one forward/backward step passed")
        return

    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    print(f"[subgoal] writing checkpoints to {run_dir}")

    best_val = float("inf")
    for epoch in range(args.max_epochs):
        train_metrics = train_one_epoch(
            lewm,
            predictor,
            train_loader,
            optimizer,
            device,
            freeze_lewm_encoder=args.freeze_lewm_encoder,
            cosine_loss_weight=args.cosine_loss_weight,
            limit_batches=args.limit_train_batches,
        )
        val_metrics = (
            evaluate(
                lewm,
                predictor,
                val_loader,
                device,
                cosine_loss_weight=args.cosine_loss_weight,
            )
            if val_loader is not None
            else None
        )

        message = f"[subgoal] epoch {epoch + 1:03d}/{args.max_epochs} train_loss={train_metrics['loss']:.6f}"
        if val_metrics is not None:
            message += f" val_loss={val_metrics['loss']:.6f}"
        print(message)

        val_for_best = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        if val_for_best < best_val:
            best_val = val_for_best
            save_checkpoint(run_dir / "best.pt", predictor, args, embed_dim, epoch, best_val)
        save_checkpoint(run_dir / "last.pt", predictor, args, embed_dim, epoch, val_for_best)


def build_or_load_lewm(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.lewm_ckpt:
        print(f"[subgoal] loading LeWM checkpoint: {args.lewm_ckpt}")
        return load_lewm_model(
            args.lewm_ckpt,
            config_path=args.lewm_config,
            device=device,
            freeze=args.freeze_lewm_encoder,
            action_dim=7,
        )

    if args.allow_random_lewm:
        print("[subgoal] using random LeWM-shaped encoder for smoke test only")
        model = build_random_lewm_for_smoke(
            image_size=args.image_size,
            history_size=args.history_size,
            embed_dim=192,
            action_dim=7,
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model

    raise ValueError(
        "--lewm_ckpt is required for real subgoal training. "
        "Use --allow_random_lewm only for local dry-run plumbing tests."
    )


def train_one_epoch(
    lewm: torch.nn.Module,
    predictor: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    freeze_lewm_encoder: bool,
    cosine_loss_weight: float,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    predictor.train()
    if freeze_lewm_encoder:
        lewm.eval()
    else:
        lewm.train()

    totals = {"loss": 0.0, "mse_loss": 0.0, "cosine_loss": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        history_emb, future_emb = encode_batch(
            lewm, batch, device, freeze_lewm_encoder=freeze_lewm_encoder
        )
        pred_emb = predictor(history_emb)
        loss, metrics = latent_subgoal_loss(
            pred_emb, future_emb, cosine_weight=cosine_loss_weight
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()

        for key in totals:
            totals[key] += float(metrics[key])
        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    lewm: torch.nn.Module,
    predictor: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    cosine_loss_weight: float,
) -> Dict[str, float]:
    lewm.eval()
    predictor.eval()
    totals = {"loss": 0.0, "mse_loss": 0.0, "cosine_loss": 0.0}
    count = 0
    for batch in loader:
        history_emb, future_emb = encode_batch(
            lewm, batch, device, freeze_lewm_encoder=True
        )
        pred_emb = predictor(history_emb)
        _, metrics = latent_subgoal_loss(
            pred_emb, future_emb, cosine_weight=cosine_loss_weight
        )
        for key in totals:
            totals[key] += float(metrics[key])
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def encode_batch(
    lewm: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    freeze_lewm_encoder: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    history_pixels = batch["history_pixels"].to(device, non_blocking=True)
    future_pixels = batch["future_pixels"].to(device, non_blocking=True)

    if freeze_lewm_encoder:
        with torch.no_grad():
            history_emb = encode_pixels(lewm, history_pixels)
            future_emb = encode_pixels(lewm, future_pixels)
    else:
        history_emb = lewm.encode({"pixels": history_pixels})["emb"]
        with torch.no_grad():
            future_emb = encode_pixels(lewm, future_pixels)
    return history_emb, future_emb.detach()


@torch.no_grad()
def infer_embed_dim(
    lewm: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device
) -> int:
    lewm.eval()
    history_pixels = batch["history_pixels"].to(device, non_blocking=True)
    emb = encode_pixels(lewm, history_pixels)
    print(f"[subgoal] LeWM latent shape: {tuple(emb.shape)}")
    return int(emb.shape[-1])


def split_dataset(
    dataset: PiperSubgoalDataset, val_split: float, seed: int
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if len(dataset) < 2 or val_split <= 0:
        return dataset, torch.utils.data.Subset(dataset, [])
    val_len = max(1, int(round(len(dataset) * val_split)))
    val_len = min(val_len, len(dataset) - 1)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def make_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, **kwargs)


def save_checkpoint(
    path: Path,
    predictor: LatentSubgoalPredictor,
    args: argparse.Namespace,
    embed_dim: int,
    epoch: int,
    metric: float,
) -> None:
    torch.save(
        {
            "predictor_state_dict": predictor.state_dict(),
            "embed_dim": embed_dim,
            "history_size": args.history_size,
            "subgoal_offsets": parse_subgoal_offsets(args.subgoal_offsets),
            "epoch": epoch,
            "metric": metric,
            "args": vars(args),
        },
        path,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
