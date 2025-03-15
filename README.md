# Robotic Manipulation Data Generation Pipeline

This repository contains a data generation pipeline for creating pseudo-demonstrations of robotic manipulation tasks. The pipeline generates synthetic data by sampling objects from ShapeNet, creating pseudo-tasks with waypoints, and recording gripper trajectories and observations.

## Features

- Object sampling from ShapeNet dataset
- Random waypoint generation for manipulation tasks
- Multiple interpolation strategies (linear, cubic, spherical)
- Gripper state changes for grasping simulation
- Multi-view depth camera observations using PyRender
- Uniform trajectory sampling (1cm spatial, 3° angular resolution)
- Support for generating multiple demonstrations per task

## Requirements

```
numpy>=1.21.0
trimesh>=3.9.0
pyrender>=0.1.45
scipy>=1.7.0
transforms3d>=0.3.1
shapenet-utils>=0.3.0
open3d>=0.13.0
pytorch3d>=0.6.0
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the ShapeNet dataset and Robotiq 2F-85 gripper mesh

## Usage

The main script for generating data is `data_generation/generate_data.py`. Basic usage:

```bash
python data_generation/generate_data.py \
    --shapenet_path /path/to/shapenet \
    --gripper_mesh_path /path/to/robotiq_2f85.obj \
    --output_path /path/to/output \
    --num_demos 1000 \
    --demos_per_task 5
```

### Arguments

- `--shapenet_path`: Path to ShapeNet dataset
- `--gripper_mesh_path`: Path to Robotiq 2F-85 gripper mesh file
- `--output_path`: Directory to save generated data
- `--num_demos`: Total number of demonstrations to generate
- `--demos_per_task`: Number of demonstrations per unique task

## Output Format

The generated data is organized as follows:

```
output_path/
├── demo_000000/
│   ├── trajectory.npy          # Gripper poses and states
│   ├── metadata.npy           # Task metadata
│   └── observations/
│       ├── frame_000000/
│       │   ├── camera_0_color.npy
│       │   └── camera_0_depth.npy
│       └── ...
├── demo_000001/
└── ...
```

### Data Format

- `trajectory.npy`: Dictionary containing
  - `poses`: List of 4x4 transformation matrices
  - `states`: List of boolean gripper states (open/closed)
  
- `metadata.npy`: Dictionary containing
  - `demo_id`: Unique demonstration identifier
  - `object_transforms`: Initial object poses
  - `waypoints`: List of waypoint poses
  - `gripper_states`: List of gripper states at waypoints

- Observations:
  - `color`: RGB image array (480x640x3)
  - `depth`: Depth image array (480x640)

## Implementation Details

The implementation is split into three main components:

1. `SceneGenerator`: Handles object sampling, placement, and waypoint generation
2. `TrajectoryGenerator`: Implements different interpolation strategies
3. `DataGenerator`: Orchestrates the overall data generation process

### Interpolation Methods

- **Linear**: Direct linear interpolation between waypoints
- **Cubic**: Cubic spline interpolation for position, SLERP for rotation
- **Spherical**: Interpolation while maintaining constant distance from origin

## Notes

- Generated trajectories are not checked for dynamic or kinematic feasibility
- Object placement is random but constrained to a plane
- Gripper state changes are assigned to random waypoints
- Camera view is fixed at a single position above and behind the scene 