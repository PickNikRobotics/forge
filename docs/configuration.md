# Configuration Guide

Forge supports YAML configuration files for customizing conversions. This is useful when:

- Your dataset has non-standard field names
- You want to rename cameras in the output
- You need to apply transforms (e.g., normalizing gripper values)
- You're converting multiple datasets with the same settings

## Quick Start

1. **Generate a config template** from your dataset:
   ```bash
   forge inspect my_dataset/ --generate-config config.yaml
   ```

2. **Edit the config** to customize mappings:
   ```yaml
   target_format: lerobot-v3
   fps: 30
   robot_type: franka

   cameras:
     wrist_camera: wrist_cam
     overhead_camera: front_cam

   fields:
     action: data/actions
     state: data/proprio
   ```

3. **Run the conversion**:
   ```bash
   forge convert my_dataset/ output/ --config config.yaml
   ```

## Configuration Reference

### Basic Settings

```yaml
# Target format (required)
target_format: lerobot-v3

# Source format (auto-detected if not specified)
source_format: rlds

# Frames per second (required if not in source metadata)
fps: 30

# Robot identifier
robot_type: franka
```

### Camera Mapping

Map source camera names to target names. Forge uses fuzzy matching, so you don't need to include common prefixes like `steps/observation/`.

```yaml
cameras:
  # source_name: target_name
  agentview_image: front_cam
  robot0_eye_in_hand_image: wrist_cam

  # These all match the same source camera:
  # - steps/observation/wrist_camera
  # - observation.images.wrist_camera
  # - wrist_camera
  wrist_camera: wrist_cam
```

### Field Mapping

Map source field paths to standardized names:

```yaml
fields:
  # Simple mapping (source path)
  action: steps/action
  state: observation/robot_state

  # With explicit target name
  gripper:
    source: observation/gripper_pos
    target: gripper_state

  # With transform (future feature)
  gripper_binary:
    source: observation/gripper_pos
    target: gripper_state
    transform: binary  # Normalize to 0/1
```

### Video Settings

```yaml
video:
  codec: h264          # Video codec (default: h264)
  crf: 23              # Quality: lower = better, 18-28 typical
  compress: true       # Enable compression
```

### Writer Settings

Options passed straight to the target format's writer config (unknown keys are
ignored):

```yaml
writer_config:
  video: false         # lerobot-v3: store camera frames as PNG bytes inline in
                       # the data parquet (dtype "image") instead of MP4 videos.
                       # Default true. Needs the [lerobot] extra (Pillow).
```

### Behavior Settings

```yaml
fail_on_error: false   # Stop on first error vs continue
skip_existing: true    # Skip episodes that already exist in output
include_depth: true    # Include depth camera streams
```

### Performance Settings

```yaml
num_workers: 4         # Parallel workers for episode processing (default: 1)
```

Using multiple workers can significantly speed up conversion (10-30x on multi-core systems). Each worker processes episodes independently using ProcessPoolExecutor.

## CLI Override

Command-line arguments override config file settings:

```bash
# Config says fps: 30, but this overrides to 60
forge convert input/ output/ --config config.yaml --fps 60

# Add additional camera mappings
forge convert input/ output/ --config config.yaml --camera extra_cam=side_cam

# Use 4 parallel workers for faster conversion
forge convert input/ output/ --config config.yaml --workers 4
```

## Example Configs

### RLDS (robomimic) to LeRobot v3

```yaml
target_format: lerobot-v3
fps: 20
robot_type: panda

cameras:
  agentview_image: front_cam
  robot0_eye_in_hand_image: wrist_cam

fields:
  action: steps/action
  state: steps/observation/robot0_eef_pos
```

### Zarr (Diffusion Policy) to LeRobot v3

```yaml
target_format: lerobot-v3
fps: 10
robot_type: pusht

cameras:
  img: observation_image

fields:
  action: action
  state: state
```

### Custom Dataset

```yaml
target_format: lerobot-v3
fps: 30
robot_type: franka

cameras:
  cam_wrist: wrist_cam
  cam_front: front_cam
  cam_side: side_cam

fields:
  action: robot/commanded_joint_positions
  state:
    source: robot/measured_joint_positions
    target: state
  gripper:
    source: robot/gripper_position
    target: gripper_state

video:
  crf: 20  # Higher quality

fail_on_error: true  # Stop on any error
```

## Dry Run

Preview what a conversion will do without writing files:

```bash
forge convert input/ output/ --config config.yaml --dry-run
```

Output shows:
- Detected source format and schema
- Camera mappings that will be applied
- Field mappings configured
- Target format and settings
