# le-wm

LeWM adapted for PiperX LeRobot training and real-robot deployment.

This repository started from the LeWorldModel codebase, but the checkout now
contains the practical pieces needed to train and deploy a left-arm Piper policy:

- a JEPA-style latent world model in `jepa.py`, `module.py`, and `train.py`
- LeRobot/Piper dataset loaders in `piper_lerobot_dataset.py`
- a latent subgoal predictor in `subgoal_predictor.py` and
  `train_subgoal_predictor.py`
- an HTTP policy server that combines the LeWM checkpoint, subgoal predictor,
  and CEM action planner
- a repo-owned ROS Noetic runtime for Piper control, USB camera observation,
  mux switching, and the real-robot policy bridge
- deployment scripts and a terminal UI under `scripts/`

The current real deployment target is the left Piper follower arm with one USB
observation camera trained from `observation.image`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `jepa.py` | LeWM/JEPA model: image encoder, action-conditioned latent rollout, latent cost |
| `module.py` | Transformer blocks, action embedder, autoregressive predictor, SIGReg |
| `train.py` | Lightning/Hydra training entrypoint for the LeWM world model |
| `piper_lerobot_dataset.py` | LeRobot dataset adapter for Piper demonstrations |
| `piper_subgoal_dataset.py` | Dataset windows for latent subgoal training |
| `subgoal_predictor.py` | History-to-future-latent subgoal predictor |
| `train_subgoal_predictor.py` | Subgoal predictor training entrypoint |
| `scripts/run_lewm_subgoal_policy_server.py` | HTTP inference server and CEM planner |
| `scripts/run_single_arm_policy_bridge_ros.py` | ROS bridge from camera/state to policy server and joint commands |
| `scripts/run_piper_left_top_policy_real.sh` | Combined policy server, mux switch, and real bridge launcher |
| `scripts/run_piper_left_top_robot_independent.sh` | Minimal robot + camera ROS stack |
| `scripts/lewm_deploy_tui.py` | Optional TUI for build, dry-run, mux, and real deployment |
| `ros_ws/` | Repo-owned Catkin workspace for Piper messages, controller, mux, camera nodes |
| `vendor/` | Vendored Python dependencies needed by the ROS runtime |
| `config/` | Hydra configs and local deployment pose config |
| `inference_view_logs/` | Runtime CSV logs and optional debug images |

## Model and deployment loop

At deployment time the bridge samples robot state and camera frames at the
policy history rate, sends a dense history payload to the HTTP server, receives
an action chunk, smooths/guards it, and publishes to the selected ROS mux input.

The server loop is:

1. encode the recent camera/action history with the LeWM model
2. predict latent future subgoals from that history
3. sample candidate action chunks with CEM
4. score candidates by latent world-model rollout cost plus smoothness terms
5. return the selected action sequence to the ROS bridge

The bridge also supports ACT-style temporal ensembling of overlapping action
chunks. The current deployment path sends dense 30 Hz history payloads and
reports how many policy steps were actually executed so the server warm-starts
the next CEM solve at the correct point in the previous plan.

Useful runtime checks in the bridge log:

```text
history=payload/10 warm_shift=8
```

For the current settings, `history=payload/10` means the server is using the
dense 10-frame history from the bridge, and `warm_shift=8` after the first chunk
means the CEM warm start is aligned with the 8 executed policy steps.

## Setup

Create the Python environment:

```bash
cd /home/pairlab/le-wm
scripts/vast_setup.sh
source .venv/bin/activate
```

Build the repo-owned Catkin workspace once before ROS deployment:

```bash
cd /home/pairlab/le-wm
scripts/build_lewm_ros_ws.sh
```

Download the dataset and checkpoints when needed:

```bash
cd /home/pairlab/le-wm
hf download elysia111/piperx-lerobot-static-eraser-pnp \
  --repo-type dataset \
  --local-dir ./lerobot_data/static_eraser_pnp_v30_fps30

hf download elysia111/piperx-lewm-checkpoints \
  --local-dir ./checkpoints_from_hf
```

The default dataset path used by the scripts is:

```text
./lerobot_data/static_eraser_pnp_v30_fps30
```

## Training

Train the LeWM world model on the Piper LeRobot dataset:

```bash
cd /home/pairlab/le-wm
source .venv/bin/activate

PIPERX_LEROBOT_ROOT=./lerobot_data/static_eraser_pnp_v30_fps30 \
MAX_EPOCHS=100 \
BATCH_SIZE=32 \
NUM_WORKERS=4 \
scripts/train_piperx_lewm.sh
```

Train the latent subgoal predictor from a trained LeWM checkpoint:

```bash
cd /home/pairlab/le-wm
source .venv/bin/activate

python train_subgoal_predictor.py \
  --lewm_ckpt checkpoints_from_hf/lewm/weights_epoch_300.pt \
  --data_root lerobot_data/static_eraser_pnp_v30_fps30 \
  --history_size 10 \
  --subgoal_offsets 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --max_epochs 50 \
  --batch_size 16
```

For a one-batch plumbing check:

```bash
python train_subgoal_predictor.py \
  --allow_random_lewm \
  --dry_run
```

## Deployment

Use two terminals for the current real-robot path. Terminal 1 owns the ROS
container, robot controller, mux, and camera. Terminal 2 starts or reuses the
policy server, switches the mux, and starts the bridge.

Current local hardware defaults:

| Setting | Value |
| --- | --- |
| ROS master | `http://localhost:11411` |
| Left follower CAN | `can1` |
| Observation camera USB port | `3-4` |
| Policy endpoint | `http://127.0.0.1:<PORT>/infer` |

Bring the CAN interface up before starting the robot stack if needed:

```bash
ip link show can1
sudo ip link set can1 up type can bitrate 1000000
```

Terminal 1, robot and camera ROS stack:

```bash
cd /home/pairlab/le-wm

ROS_MASTER_URI=http://localhost:11411 \
LEFT_CAN_PORT=can1 \
CAMERA_OBS_USB_PORT=3-4 \
scripts/run_piper_left_top_robot_independent.sh
```

Terminal 2, policy server plus real bridge:

```bash
cd /home/pairlab/le-wm

PORT=8768 \
SERVER_URL=http://127.0.0.1:8768/infer \
ROS_MASTER_URI=http://localhost:11411 \
LEWM_POLICY_HISTORY_SIZE=10 \
LEWM_POLICY_HISTORY_SAMPLE_RATE=30 \
LEWM_PREPOSITION_TARGET_FILE=config/lewm_left_avg_start_pose.json \
LEWM_WAIT_ENTER_AFTER_PREPOSITION=1 \
LEWM_NUM_SAMPLES=128 \
LEWM_TOPK=16 \
LEWM_CEM_STEPS=4 \
LEWM_INIT_STD=0.45 \
LEWM_MAX_JOINT_STEP_RAD=0.035 \
LEWM_EXECUTE_HORIZON=15 \
LEWM_ACTION_TEMPORAL_ENSEMBLE=1 \
LEWM_ACTION_TEMPORAL_ENSEMBLE_HISTORY=8 \
LEWM_ACTION_TEMPORAL_ENSEMBLE_EXECUTE_STEPS=8 \
LEWM_ACTION_CHUNK_SMOOTH_PASSES=4 \
LEWM_GUARD_EMA_ALPHA=0.40 \
LEWM_GUARD_MAX_JOINT_STEP=0.04 \
LEWM_ACTION_MAX_JOINT_VELOCITY_RAD_S=0.30 \
LEWM_ACTION_MAX_JOINT_ACCEL_RAD_S2=0.80 \
scripts/run_piper_left_top_policy_real.sh
```

The launcher asks for confirmation before real publishing. The bridge will
preposition, hold, and wait for Enter before inference begins. Stop Terminal 2
first, then stop Terminal 1.

## Dry run and debugging

Run the bridge without publishing real commands:

```bash
cd /home/pairlab/le-wm

ROS_MASTER_URI=http://localhost:11411 \
EXECUTE_REAL=0 \
scripts/run_piper_left_top_bridge_independent.sh
```

Save a few camera frames as seen by the bridge:

```bash
ROS_MASTER_URI=http://localhost:11411 \
LEWM_DEBUG_SAVE_IMAGE_DIR=/workspace/le-wm/inference_view_logs/debug_images \
LEWM_DEBUG_SAVE_IMAGE_LIMIT=4 \
EXECUTE_REAL=0 \
scripts/run_piper_left_top_bridge_independent.sh
```

Action chunk CSVs are written to:

```text
inference_view_logs/lewm_left_top_action_chunks_*.csv
```

Recent deployment diagnostics to check:

- `server_history_source` should be `payload`
- `server_history_payload_len` should be `10`
- `server_warm_start_shift_steps` should be `0` on the first request, then
  normally `8` with the settings above
- `published_steps` should match `expected_publish_steps` for full chunks

## TUI

The TUI wraps the same deployment steps:

```bash
cd /home/pairlab/le-wm
python3 scripts/lewm_deploy_tui.py
```

Typical order:

1. build the Catkin workspace if needed
2. start robot + observation camera
3. start the policy server
4. dry-run the bridge
5. switch the mux to policy
6. start real publishing only after the dry run is healthy

## Common failures

`Address already in use`

Use a fresh `PORT` and matching `SERVER_URL`, or stop the old policy server.

`No such container: lewm_ros_runtime`

Start Terminal 1 first. The bridge runs inside the ROS container created by the
robot stack script.

`CAN socket can_sl does not exist`

The script was using the default CAN name. Set `LEFT_CAN_PORT=can1` for the
current left follower arm.

`CAN port can1 is not UP`

Bring the interface up on the host:

```bash
sudo ip link set can1 up type can bitrate 1000000
```

Bridge logs say mux is not selected

Switch the mux to `/robot/arm_left/vla_joint_cmd`, or use
`scripts/run_piper_left_top_policy_real.sh`, which performs the mux switch
before starting the real bridge.

## Upstream

The model code is based on LeWorldModel:

- paper: `https://arxiv.org/pdf/2603.19312v1`
- website: `https://le-wm.github.io/`
- original project: `https://github.com/lucas-maes/le-wm`

This fork adds the PiperX LeRobot dataset path, subgoal-predictor workflow,
ROS/Piper deployment runtime, and real-robot bridge scripts.
