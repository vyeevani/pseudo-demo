import numpy as np
import pyrender
import trimesh
from pathlib import Path
from typing import List, Dict, Any, Optional
from scene_generator import SceneGenerator
from trajectory_generator import TrajectoryGenerator

class DataGenerator:
    def __init__(self,
                 shapenet_path: Optional[str] = None,
                 gripper_mesh_path: Optional[str] = None):
        """
        Initialize the data generator.
        
        Args:
            shapenet_path: Path to ShapeNet dataset
            gripper_mesh_path: Path to Robotiq 2F-85 gripper mesh
        """
        self.scene_generator = SceneGenerator(shapenet_path, gripper_mesh_path)
        self.trajectory_generator = TrajectoryGenerator()
        
    def generate_demonstration(self,
                             demo_id: int,
                             interpolation_method: str = 'linear'
                             ) -> Dict[str, Any]:
        """
        Generate a single demonstration.
        
        Args:
            demo_id: Unique identifier for this demonstration
            interpolation_method: Method to use for trajectory interpolation
            
        Returns:
            Dictionary containing demonstration data
        """
        # Sample objects
        objects = self.scene_generator.sample_objects(num_objects=2)
        
        # Place objects randomly on the table
        object_transforms = self.scene_generator.place_objects(objects)
        
        # Sample waypoints and touched objects
        poses, touched_objects = self.scene_generator.sample_waypoints(object_transforms)
        
        # Generate trajectory
        poses, touched_objects = self.trajectory_generator.interpolate_trajectory(
            poses,
            touched_objects,
            method=interpolation_method
        )

        # Generate object poses
        object_poses = []
        for obj_idx in range(len(objects)):
            obj_poses = self.generate_object_poses(obj_idx, object_transforms[obj_idx], poses, touched_objects)
            object_poses.append(obj_poses)
        
        # Render observations
        observations = self._render_trajectory(poses, objects, object_poses)
        
        # Return demonstration data
        demo_data = {
            'demo_id': demo_id,
            'object_transforms': object_transforms,  # Use actual object transforms
            'touched_objects': touched_objects,
            'trajectory': {
                'poses': poses,
                'touched_objects': touched_objects
            },
            'observations': observations
        }
        
        return demo_data
    
    def generate_object_poses(self, object_index: int, initial_transform: np.ndarray, 
                              gripper_points: List[np.ndarray], 
                              touched_objects: List[Optional[int]]) -> List[np.ndarray]:
        """
        Generate object poses based on gripper points and touched objects.
        
        Args:
            object_index: Index of the object to track
            initial_transform: Initial transformation matrix for the object
            gripper_points: List of gripper transformation matrices
            touched_objects: List of indices of objects being touched/grasped at each point
            
        Returns:
            List of transformation matrices representing the object's poses
        """
        object_poses = []
        # Start with the initial placement transform from the scene generator
        current_pose = initial_transform.copy()
        object_poses.append(current_pose.copy())
        
        for i in range(1, len(gripper_points)):
            # Get previous state
            is_touched_now = touched_objects[i] == object_index
            was_touched_before = touched_objects[i-1] == object_index
            
            # Calculate gripper movement from previous to current frame
            gripper_delta = gripper_points[i] @ np.linalg.inv(gripper_points[i-1])
            
            if was_touched_before and is_touched_now:
                # Object remains grasped - move with the gripper
                current_pose = gripper_delta @ current_pose
            elif not was_touched_before and is_touched_now:
                # Object just got picked up - calculate and store the grasp transform
                # This represents how the object is positioned relative to the gripper
                current_pose = gripper_points[i] @ np.linalg.inv(gripper_points[i]) @ current_pose
            elif was_touched_before and not is_touched_now:
                # Object just got released - apply one final transform from the gripper
                current_pose = gripper_delta @ current_pose
            else:
                # Object is not being touched - it stays where it is
                pass
                
            object_poses.append(current_pose.copy())
        
        return object_poses
    
    def _render_trajectory(self, poses: List[np.ndarray], objects: List[trimesh.Trimesh], object_poses: List[List[np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Render observations along the trajectory."""
        observations = []
        # Create renderer with anti-aliasing
        renderer = pyrender.OffscreenRenderer(
            viewport_width=640,
            viewport_height=480,
            point_size=1.0,
        )
        
        # Create gripper node once
        gripper_node = pyrender.Node(
            mesh=pyrender.Mesh.from_trimesh(
                self.scene_generator.gripper_mesh,
                smooth=True  # Enable smooth shading
            ),
            matrix=np.eye(4)  # Initialize with identity matrix
        )
        self.scene_generator.scene.add_node(gripper_node)
        
        # Create object nodes
        object_nodes = []
        for obj, obj_pose in zip(objects, object_poses):
            mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)
            node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            self.scene_generator.scene.add_node(node)
            object_nodes.append(node)
        
        for pose_idx, pose in enumerate(poses):
            # Update gripper pose
            self.scene_generator.scene.set_pose(gripper_node, pose)
            
            # Update object poses
            for obj_idx, obj_node in enumerate(object_nodes):
                self.scene_generator.scene.set_pose(obj_node, object_poses[obj_idx][pose_idx])
            
            # Render from all cameras with flags for better quality
            frame_observations = {}
            for i, camera in enumerate(self.scene_generator.cameras):
                flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                color, depth = renderer.render(self.scene_generator.scene, flags=flags)
                frame_observations[f'camera_{i}'] = {
                    'color': color,
                    'depth': depth
                }
            
            observations.append(frame_observations)
        
        # Clean up
        self.scene_generator.scene.remove_node(gripper_node)
        for obj_node in object_nodes:
            self.scene_generator.scene.remove_node(obj_node)
        renderer.delete()
        return observations