# PiperX Latent Subgoal Predictor

This is a separate training path from LeWM. LeWM remains the action-conditioned
world model:

```text
image history + candidate actions -> future latent states
```

The subgoal predictor learns:

```text
image history -> future latent subgoal sequence
```

It uses demonstration future frames as supervision, but the target is the frozen
LeWM image embedding of those frames, not RGB pixels and not actions. That keeps
the output in the same latent space that the CEM planner already scores against.

## Dataset

`piper_subgoal_dataset.py` reads the same Piper LeRobot export as
`piper_lerobot_dataset.py` and returns episode-local windows:

- `history_pixels`: `(history_size, 3, 224, 224)`
- `future_pixels`: `(num_subgoals, 3, 224, 224)`

With the default settings, a sample at time `t` uses history frames
`t-9 ... t` and target future frames `t+5, t+10, t+15, t+20, t+30`.

## Local Smoke Test

Use a real LeWM checkpoint when available:

```bash
source .venv/bin/activate
python train_subgoal_predictor.py \
  --dry_run \
  --lewm_ckpt /path/to/weights_epoch_300.pt \
  --data_root ./lerobot_data/static_eraser_pnp_v30_fps30 \
  --history_size 10 \
  --subgoal_offsets 5,10,15,20,30 \
  --batch_size 2 \
  --num_workers 0 \
  --device cpu
```

For a plumbing-only test on a laptop without the trained checkpoint:

```bash
python train_subgoal_predictor.py \
  --dry_run \
  --allow_random_lewm \
  --data_root ./lerobot_data/static_eraser_pnp_v30_fps30 \
  --history_size 3 \
  --subgoal_offsets 1,2 \
  --batch_size 2 \
  --num_workers 0 \
  --device cpu
```

The random-LeWM mode verifies shapes and one optimizer step only. It is not a
valid training run.

## Vast Training

After LeWM training, prefer the stable-worldmodel checkpoint produced under
`outputs/stable_worldmodel/checkpoints/lewm/weights_epoch_*.pt`:

```bash
source .venv/bin/activate
python train_subgoal_predictor.py \
  --lewm_ckpt outputs/stable_worldmodel/checkpoints/lewm/weights_epoch_300.pt \
  --data_root ./lerobot_data/static_eraser_pnp_v30_fps30 \
  --history_size 10 \
  --subgoal_offsets 5,10,15,20,30 \
  --batch_size 16 \
  --max_epochs 50 \
  --num_workers 4 \
  --device cuda
```

Lightning `last.ckpt` loading is supported best-effort if the matching LeWM
training config can be found. If it cannot, pass `--lewm_config /path/to/config.yaml`
or use `weights_epoch_*.pt`.

## Later CEM Connection

At runtime, this predictor can produce a latent subgoal sequence from the live
history image buffer. The online CEM planner can then score candidate action
rollouts against those predicted latent subgoals instead of a fixed hand-picked
goal image. This task does not move the robot or integrate the predictor into
the ROS/CEM runtime yet.
