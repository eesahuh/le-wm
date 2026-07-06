"""Latent subgoal sequence predictor for PiperX LeWM.

The predictor is intentionally separate from the action-conditioned LeWM world
model. It maps a history of LeWM image latents to future LeWM image latents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch import nn


class LatentSubgoalPredictor(nn.Module):
    """Small Transformer encoder/decoder over LeWM latent sequences."""

    def __init__(
        self,
        *,
        embed_dim: int,
        history_size: int,
        num_subgoals: int,
        hidden_dim: Optional[int] = None,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.history_size = int(history_size)
        self.num_subgoals = int(num_subgoals)
        self.hidden_dim = int(hidden_dim or embed_dim * 4)
        self.heads = _compatible_heads(self.embed_dim, heads)

        self.input_norm = nn.LayerNorm(self.embed_dim)
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.history_pos = nn.Parameter(torch.zeros(1, self.history_size, self.embed_dim))
        self.subgoal_queries = nn.Parameter(
            torch.zeros(1, self.num_subgoals, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.subgoal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.head = nn.Linear(self.embed_dim, self.embed_dim)

        nn.init.trunc_normal_(self.history_pos, std=0.02)
        nn.init.trunc_normal_(self.subgoal_queries, std=0.02)

    def forward(self, history_emb: torch.Tensor) -> torch.Tensor:
        """Predict future latent subgoals from ``(B, H, D)`` history latents."""

        if history_emb.ndim != 3:
            raise ValueError(f"Expected history_emb shape (B, H, D), got {history_emb.shape}")
        batch_size, steps, embed_dim = history_emb.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")
        if steps > self.history_size:
            raise ValueError(
                f"history_emb has {steps} steps, but predictor was built for "
                f"history_size={self.history_size}"
            )

        history = self.input_norm(history_emb)
        history = history + self.history_pos[:, :steps]
        memory = self.history_encoder(history)
        queries = self.subgoal_queries.expand(batch_size, -1, -1)
        decoded = self.subgoal_decoder(queries, memory)
        return self.head(self.output_norm(decoded))


@torch.no_grad()
def encode_pixels(lewm_model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Encode image sequences with the LeWM image encoder."""

    output = lewm_model.encode({"pixels": pixels})
    return output["emb"]


def latent_subgoal_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    *,
    cosine_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MSE loss, with optional cosine-distance shaping."""

    target = target_emb.detach()
    mse = F.mse_loss(pred_emb, target)
    cosine = pred_emb.new_tensor(0.0)
    if cosine_weight > 0:
        cosine = 1.0 - F.cosine_similarity(
            pred_emb.flatten(0, 1), target.flatten(0, 1), dim=-1
        ).mean()
    loss = mse + float(cosine_weight) * cosine
    return loss, {"loss": loss.detach(), "mse_loss": mse.detach(), "cosine_loss": cosine.detach()}


def load_lewm_model(
    checkpoint: Union[str, Path],
    *,
    config_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    action_dim: int = 7,
) -> nn.Module:
    """Load a trained LeWM model from a stable-worldmodel ``.pt`` or Lightning ``.ckpt``.

    The stable-worldmodel ``weights_epoch_*.pt`` file is the preferred format.
    Lightning ``.ckpt`` support is best-effort and requires a saved training
    config with the model section.
    """

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LeWM checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".pt" or checkpoint_path.is_dir():
        model = _load_stable_worldmodel_checkpoint(checkpoint_path)
    elif checkpoint_path.suffix == ".ckpt":
        model = _load_lightning_checkpoint(
            checkpoint_path, config_path=config_path, action_dim=action_dim
        )
    else:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_path.suffix!r}; "
            "use weights_epoch_*.pt or last.ckpt"
        )

    model.to(device)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model


def build_random_lewm_for_smoke(
    *,
    image_size: int = 224,
    history_size: int = 10,
    embed_dim: int = 192,
    action_dim: int = 7,
) -> nn.Module:
    """Build an untrained LeWM-shaped model for local smoke tests only."""

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    encoder = vit_hf(
        size="tiny",
        patch_size=14,
        image_size=image_size,
        pretrained=False,
        use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(input_dim=action_dim, emb_dim=embed_dim),
        projector=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        ),
    )


def _load_stable_worldmodel_checkpoint(checkpoint_path: Path) -> nn.Module:
    import stable_worldmodel as swm

    # stable_worldmodel resolves relative paths under ~/.stable_worldmodel/checkpoints.
    # Resolve first so repo-local checkpoint paths keep pointing at the repo.
    return swm.wm.utils.load_pretrained(str(checkpoint_path.expanduser().resolve()))


def _load_lightning_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Optional[Union[str, Path]],
    action_dim: int,
) -> nn.Module:
    model_config_path = _find_lewm_config(checkpoint_path, config_path)
    model = _instantiate_from_config(model_config_path, action_dim=action_dim)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = _extract_lewm_state_dict(state_dict)
    if not model_state:
        raise RuntimeError(f"No LeWM model weights found in {checkpoint_path}")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    total_keys = len(model.state_dict())
    if len(missing) > total_keys // 2:
        raise RuntimeError(
            "Lightning checkpoint did not match the LeWM model config. "
            f"Missing {len(missing)} of {total_keys} keys. "
            f"Unexpected keys: {unexpected[:5]}"
        )
    if unexpected:
        print(f"[subgoal] ignored {len(unexpected)} unexpected checkpoint keys")
    if missing:
        print(f"[subgoal] loaded Lightning checkpoint with {len(missing)} missing keys")
    return model


def _instantiate_from_config(config_path: Path, *, action_dim: int) -> nn.Module:
    import hydra

    cfg = OmegaConf.load(config_path)
    if "model" not in cfg:
        raise KeyError(f"Config {config_path} does not contain a 'model' section")

    with open_dict(cfg):
        if OmegaConf.is_missing(cfg.model.action_encoder, "input_dim"):
            cfg.model.action_encoder.input_dim = action_dim

    return hydra.utils.instantiate(cfg.model)


def _extract_lewm_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("encoder.", "predictor.", "action_encoder.", "projector.", "pred_proj.")
    model_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        candidates: Iterable[str] = (
            key,
            key.removeprefix("model."),
            key.removeprefix("module.model."),
            key.removeprefix("_orig_mod.model."),
        )
        for candidate in candidates:
            if candidate.startswith(prefixes):
                model_state[candidate] = value
                break
    return model_state


def _find_lewm_config(
    checkpoint_path: Path, config_path: Optional[Union[str, Path]]
) -> Path:
    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LeWM config not found: {path}")
        return path

    candidates = []
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        candidates.extend([parent / "config.yaml", parent / ".hydra" / "config.yaml"])

    repo_default = Path(__file__).resolve().parent / "outputs" / "stable_worldmodel" / "checkpoints" / "config.yaml"
    candidates.append(repo_default)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a LeWM training config for the Lightning checkpoint. "
        "Pass --lewm_config /path/to/config.yaml, or use the stable-worldmodel "
        "weights_epoch_*.pt checkpoint instead."
    )


def _compatible_heads(embed_dim: int, requested_heads: int) -> int:
    for heads in (requested_heads, 8, 4, 2, 1):
        if heads > 0 and embed_dim % heads == 0:
            return heads
    return 1
