# Vast + Hugging Face Workflow

This directory contains a minimal scripts-based workflow for using:

- GitHub for code
- Hugging Face for dataset and checkpoint transfer
- Vast.ai for GPU training

No Docker setup is required.

## Laptop

```bash
source .venv/bin/activate
huggingface-cli login
scripts/hf_upload_dataset.sh
git push
```

If your installed Hugging Face CLI says `huggingface-cli` is deprecated, use:

```bash
hf auth login
scripts/hf_upload_dataset.sh
```

Dataset upload defaults:

```bash
HF_DATASET_REPO=elysia111/piperx-lerobot-static-eraser-pnp
PIPERX_LEROBOT_ROOT=./lerobot_data/static_eraser_pnp_v30_fps30
```

## Vast

```bash
git clone <repo>
cd le-wm
scripts/vast_setup.sh
huggingface-cli login
scripts/hf_download_dataset.sh
scripts/train_piperx_lewm.sh
scripts/hf_upload_checkpoints.sh
```

If needed, use `hf auth login` instead of `huggingface-cli login`.

Training defaults:

```bash
PIPERX_LEROBOT_ROOT=./lerobot_data/static_eraser_pnp_v30_fps30
ACCELERATOR=gpu
DEVICES=1
MAX_EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=4
```

Override them inline:

```bash
MAX_EPOCHS=20 BATCH_SIZE=16 NUM_WORKERS=8 scripts/train_piperx_lewm.sh
```

You can append extra Hydra overrides after the script name:

```bash
scripts/train_piperx_lewm.sh trainer.precision=bf16-mixed
```

## Laptop After Training

```bash
huggingface-cli download elysia111/piperx-lewm-checkpoints --local-dir ./checkpoints_from_hf
```

Or with the newer CLI:

```bash
hf download elysia111/piperx-lewm-checkpoints --local-dir ./checkpoints_from_hf
```

## Notes

- Do not commit dataset files.
- Do not commit `.venv/`, `outputs/`, or downloaded checkpoints.
- `scripts/vast_setup.sh` does not force reinstall CUDA PyTorch. If the base Vast image already has CUDA-enabled PyTorch, the script creates the venv with system site packages so pip can reuse it.
- `scripts/hf_download_dataset.sh` verifies that the downloaded dataset contains:
  - `meta/info.json`
  - `data/chunk-000/file-000.parquet`
  - `videos/observation.image/chunk-000/file-000.mp4`
