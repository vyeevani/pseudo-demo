import rerun as rr
import argparse
import numpy as np
from generate_data import DataGenerator

def visualize_demonstration(shapenet_path: str, gripper_mesh_path: str):
    """Generate and visualize color and depth outputs of a demonstration using Rerun."""
    
    # Initialize Rerun
    rr.init("Robot Manipulation Demo", spawn=True)
    
    # Initialize data generator
    generator = DataGenerator(shapenet_path, gripper_mesh_path)
    
    # Generate a single demonstration
    demo_data = generator.generate_demonstration(demo_id=0, interpolation_method='linear')
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis
    for camera_id, camera_data in demo_data['observations'][0].items():
        rr.log(
            f"world/{camera_id}",
            rr.Pinhole(
                image_from_camera=camera_data['camera_intrinsics']['matrix'],
                width=camera_data['color'].shape[1],
                height=camera_data['color'].shape[0],
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
            static=True,
        )
    for i, obs in enumerate(demo_data['observations']):
        for camera_id, camera_data in obs.items():
            rr.log(f"world/{camera_id}/color", rr.Image(camera_data['color']))
            rr.log(f"world/{camera_id}/depth", rr.DepthImage(camera_data['depth']))

            rr.log(f"world/{camera_id}", rr.Transform3D(
                mat3x3=camera_data['camera_pose'][:3, :3],
                translation=camera_data['camera_pose'][:3, 3],
            ))
            
            # Add point cloud visualization
            points = camera_data['point_cloud']
            colors = camera_data['color'][..., :3].reshape(-1, 3) / 255.0
            rr.log(f"world/point_clouds/{camera_id}", rr.Points3D(positions=points, colors=colors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and visualize a robot manipulation demonstration')
    parser.add_argument('--shapenet_path', type=str, required=False, default=None,
                       help='Path to ShapeNet dataset')
    parser.add_argument('--gripper_mesh_path', type=str, required=False, default=None,
                       help='Path to Robotiq 2F-85 gripper mesh')
    
    args = parser.parse_args()
    
    visualize_demonstration(args.shapenet_path, args.gripper_mesh_path) 