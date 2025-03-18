import os
import numpy as np
import polars as pl
from PIL import Image
import trimesh
from generate_data import DataGenerator

def make_dataset(output_dir: str, num_demos: int, shapenet_path: str, gripper_mesh_path: str):
    """
    Generate a dataset of demonstrations and save to the specified directory as a Parquet file.
    
    Args:
        output_dir: Directory to save the dataset
        num_demos: Number of demonstrations to generate
        shapenet_path: Path to ShapeNet dataset
        gripper_mesh_path: Path to Robotiq 2F-85 gripper mesh
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data generator
    generator = DataGenerator(shapenet_path, gripper_mesh_path)
    
    # List to store all demonstration data
    all_demo_data = []
    
    for demo_id in range(num_demos):
        demo_data = generator.generate_demonstration(demo_id=demo_id, interpolation_method='linear')
        
        # Create a directory for each demonstration
        demo_dir = os.path.join(output_dir, f"demo_{demo_id:06d}")
        os.makedirs(demo_dir, exist_ok=True)
        
        # Save images and point clouds to files and update paths in demo_data
        for i, obs in enumerate(demo_data['observations']):
            frame_dir = os.path.join(demo_dir, f"frame_{i:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            image_dir = os.path.join(frame_dir, "image")
            pointcloud_dir = os.path.join(frame_dir, "pointcloud")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(pointcloud_dir, exist_ok=True)
            
            for camera_id, camera_data in obs.items():
                color_path = os.path.join(image_dir, f'{camera_id}_color.png')
                depth_path = os.path.join(image_dir, f'{camera_id}_depth.png')
                point_cloud_path = os.path.join(pointcloud_dir, f'{camera_id}_point_cloud.pcd')
                
                # Save color and depth images as PNG using PIL
                Image.fromarray(camera_data['color']).save(color_path)
                Image.fromarray(camera_data['depth']).save(depth_path)
                
                # Save point cloud as PCD using trimesh
                points = camera_data['point_cloud']
                point_cloud = trimesh.PointCloud(points)
                point_cloud.export(point_cloud_path, file_type='pcd')
                
                # Update paths in demo_data
                camera_data['color'] = color_path
                camera_data['depth'] = depth_path
                camera_data['point_cloud'] = point_cloud_path
        
        all_demo_data.append(demo_data)
    
    # Convert the list of dictionaries to a Polars DataFrame
    df = pl.DataFrame(all_demo_data)
    
    # Save the DataFrame as a Parquet file
    df.write_parquet(os.path.join(output_dir, "demonstrations.parquet"))
        
if __name__ == "__main__":
    output_dir = "output"
    num_demos = 10
    shapenet_path = "path_to_shapenet"
    gripper_mesh_path = "path_to_gripper_mesh"
    
    make_dataset(output_dir, num_demos, shapenet_path, gripper_mesh_path)
