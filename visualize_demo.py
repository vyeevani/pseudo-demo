import rerun as rr
import argparse
from generate_data import DataGenerator

def visualize_demonstration(shapenet_path: str, gripper_mesh_path: str):
    """Generate and visualize color outputs of a demonstration using Rerun."""
    
    # Initialize Rerun
    rr.init("Robot Manipulation Demo", spawn=True)
    
    # Initialize data generator
    generator = DataGenerator(shapenet_path, gripper_mesh_path)
    
    # Generate a single demonstration
    demo_data = generator.generate_demonstration(demo_id=0, interpolation_method='linear')
    
    # Visualize color observations
    for i, obs in enumerate(demo_data['observations']):
        for camera_id, camera_data in obs.items():
            rr.log(f"{camera_id}", rr.Image(camera_data['color']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and visualize a robot manipulation demonstration')
    parser.add_argument('--shapenet_path', type=str, required=False, default=None,
                       help='Path to ShapeNet dataset')
    parser.add_argument('--gripper_mesh_path', type=str, required=False, default=None,
                       help='Path to Robotiq 2F-85 gripper mesh')
    
    args = parser.parse_args()
    
    visualize_demonstration(args.shapenet_path, args.gripper_mesh_path) 