import numpy as np
import trimesh
import pyrender
import transforms3d as t3d
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class SceneGenerator:
    def __init__(self, shapenet_path: Optional[str] = None, gripper_mesh_path: Optional[str] = None):
        """
        Initialize the scene generator.
        
        Args:
            shapenet_path: Path to ShapeNet dataset
            gripper_mesh_path: Path to Robotiq 2F-85 gripper mesh
        """
        self.shapenet_path = Path(shapenet_path) if shapenet_path else None
        
        # Create gripper mesh with red color
        self.gripper_mesh = trimesh.load(gripper_mesh_path) if gripper_mesh_path else trimesh.creation.box(extents=[0.08, 0.08, 0.08])
        self.gripper_mesh.visual.vertex_colors = [255, 0, 0, 255]  # Red color
        
        # Create scene with light gray background
        self.scene = pyrender.Scene(bg_color=[0.9, 0.9, 0.9, 1.0])
        self.cameras = []
        self._setup_scene()
        
        # Define colors for objects
        self.object_colors = [
            [0, 0, 255, 255],  # Blue
            [0, 255, 0, 255],  # Green
        ]
        
    def _setup_scene(self):
        """Setup scene with cameras, lighting, and table."""
        # Add table mesh (light brown color)
        table_mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
        table_mesh.visual.vertex_colors = [210, 180, 140, 255]  # Light brown
        table_node = pyrender.Node(
            mesh=pyrender.Mesh.from_trimesh(table_mesh, smooth=False),
            matrix=np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.01],  # Slightly below origin
                [0.0, 0.0, 0.0, 1.0]
            ])
        )
        self.scene.add_node(table_node)

        # Add directional lights from different angles for even illumination
        direc_light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(np.pi/4), -np.sin(np.pi/4), 0.0],
            [0.0, np.sin(np.pi/4), np.cos(np.pi/4), 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.add(direc_light1, pose=light_pose1)
        
        # Add second directional light from opposite angle
        direc_light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        light_pose2 = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(-np.pi/4), -np.sin(-np.pi/4), 0.0],
            [0.0, np.sin(-np.pi/4), np.cos(-np.pi/4), 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.add(direc_light2, pose=light_pose2)
        
        # Add point light above the scene
        point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        light_pose3 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.add(point_light, pose=light_pose3)
        
        self._setup_cameras()
        
    def _create_look_at_matrix(self, camera_pos: np.ndarray, target_pos: np.ndarray = np.array([0, 0, 0]), up: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
        """Create a camera transformation matrix that looks at a target point."""
        # Normalize vectors
        forward = target_pos - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create rotation matrix
        R = np.column_stack([right, up, -forward])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = camera_pos

        return T

    def _generate_random_camera_pose(self, min_dist: float = 1.0, max_dist: float = 2.0) -> np.ndarray:
        """Generate a random camera pose that looks at the origin."""
        # Generate random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)  # azimuth
        phi = np.random.uniform(np.pi/6, np.pi/2)  # elevation (avoid too low angles)
        r = np.random.uniform(min_dist, max_dist)  # distance
        
        # Convert to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        camera_pos = np.array([x, y, z])
        return self._create_look_at_matrix(camera_pos)

    def _setup_cameras(self):
        """Setup multiple camera nodes but keep only one active at a time."""
        # Define camera intrinsics
        self.camera_width = 640
        self.camera_height = 480
        yfov = np.pi / 4.0
        aspect_ratio = 4.0/3.0
        
        # Calculate focal length from FoV
        self.fx = self.camera_width / (2 * np.tan(yfov * aspect_ratio / 2))
        self.fy = self.camera_height / (2 * np.tan(yfov / 2))
        self.cx = self.camera_width / 2
        self.cy = self.camera_height / 2
        
        # Create and store camera nodes (but don't add them to scene yet)
        self.camera_nodes = []
        self.active_camera_idx = None
        
        # Create main camera node centered above and behind the scene
        main_camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.866, -0.5, -1.0],
            [0.0, 0.5, 0.866, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        main_camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
        main_camera_node = pyrender.Node(camera=main_camera, matrix=main_camera_pose)
        self.camera_nodes.append(main_camera_node)
        
        # Create three random camera nodes
        for _ in range(3):
            random_pose = self._generate_random_camera_pose()
            camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
            camera_node = pyrender.Node(camera=camera, matrix=random_pose)
            self.camera_nodes.append(camera_node)
            
    def set_active_camera(self, camera_idx: int) -> None:
        """Set the active camera by index.
        
        Args:
            camera_idx: Index of the camera to activate
        """
        if camera_idx < 0 or camera_idx >= len(self.camera_nodes):
            raise ValueError(f"Invalid camera index {camera_idx}")
            
        # If there's an active camera, remove it
        if self.active_camera_idx is not None:
            self.scene.remove_node(self.camera_nodes[self.active_camera_idx])
            
        # Add the new camera node
        self.scene.add_node(self.camera_nodes[camera_idx])
        self.active_camera_idx = camera_idx
        
    def get_camera_pose(self, camera_idx: int) -> np.ndarray:
        """Get the pose of a specific camera.
        
        Args:
            camera_idx: Index of the camera to get pose for
            
        Returns:
            4x4 camera pose matrix
        """
        if camera_idx < 0 or camera_idx >= len(self.camera_nodes):
            raise ValueError(f"Invalid camera index {camera_idx}")
        return self.camera_nodes[camera_idx].matrix
    
    def get_camera_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsic parameters.
        
        Returns:
            Dictionary containing camera intrinsics (fx, fy, cx, cy, matrix)
        """
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.camera_width,
            'height': self.camera_height,
            'matrix': np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
        }
    
    def sample_objects(self, num_objects: int = 2) -> List[trimesh.Trimesh]:
        """
        Sample objects from ShapeNet dataset.
        
        Args:
            num_objects: Number of objects to sample
            
        Returns:
            List of loaded object meshes
        """
        # TODO: Implement actual ShapeNet sampling
        # This is a placeholder that should be replaced with actual ShapeNet loading
        objects = []
        for i in range(num_objects):
            # Create smaller boxes for testing
            obj = trimesh.creation.box(extents=[0.08, 0.08, 0.08])
            # Assign color based on index
            color_idx = min(i, len(self.object_colors) - 1)
            obj.visual.vertex_colors = self.object_colors[color_idx]
            objects.append(obj)
        return objects
    
    def place_objects(self, objects: List[trimesh.Trimesh]) -> List[np.ndarray]:
        """
        Place objects randomly on the plane.
        
        Args:
            objects: List of object meshes to place
            
        Returns:
            List of transformation matrices for placed objects
        """
        transforms = []
        for obj in objects:
            # Random position on the plane
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            z = obj.bounding_box.extents[2] / 2  # Place on plane
            
            # Random rotation around z-axis
            angle = np.random.uniform(0, 2 * np.pi)
            R = t3d.euler.euler2mat(0, 0, angle)
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            transforms.append(T)
            
        return transforms
    
    def sample_waypoints(self, 
                        object_transforms: List[np.ndarray],
                        min_points: int = 2,
                        max_points: int = 6) -> Tuple[List[np.ndarray], List[Optional[int]]]:
        """
        Sample waypoints for the gripper trajectory.
        
        Args:
            object_transforms: List of object transformation matrices
            min_points: Minimum number of waypoints
            max_points: Maximum number of waypoints
            
        Returns:
            Tuple of (waypoint poses, touched_objects)
            touched_objects: List where each entry is the index of the object being 
                             touched at that waypoint, or None if no object is being touched
        """
        waypoints = []
        touched_objects = []
        
        # Initial gripper position
        initial_pos = np.array([0.0, 0.0, 0.5])  # Example initial position
        T = np.eye(4)
        T[:3, 3] = initial_pos
        waypoints.append(T)
        touched_objects.append(None)  # Not touching any object
        
        # Phase 1: Go to the object with a random orientation
        obj_idx = np.random.randint(0, len(object_transforms))
        obj_transform = object_transforms[obj_idx]
        
        # Sample random orientation
        random_orientation = t3d.euler.euler2mat(
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi)
        )
        
        T = np.eye(4)
        T[:3, :3] = random_orientation
        T[:3, 3] = obj_transform[:3, 3] + np.random.uniform(-0.05, 0.05, 3)  # Go approximately to the object's position with some variation
        
        waypoints.append(T)
        touched_objects.append(obj_idx)  # Touching the object
        
        # Phase 2: Move to a random position above the table
        random_pos = np.random.uniform(-0.3, 0.3, 3)
        random_pos[2] = np.random.uniform(0.2, 0.5)  # Ensure it's above the table
        
        T = np.eye(4)
        T[:3, 3] = random_pos
        
        waypoints.append(T)
        touched_objects.append(None)  # Not touching any object
        
        return waypoints, touched_objects
    
    def _save_demonstration(self, demo_data: Dict[str, Any]) -> None:
        """Save demonstration data to disk."""
        if self.output_path is None:
            return
            
        demo_path = self.output_path / f"demo_{demo_data['demo_id']:06d}"
        demo_path.mkdir(exist_ok=True)
        
        # Save trajectory data
        np.save(
            demo_path / 'trajectory.npy',
            {
                'poses': demo_data['trajectory']['poses'],
                'states': demo_data['trajectory']['states'],
                'touched_objects': demo_data['trajectory'].get('touched_objects', [None] * len(demo_data['trajectory']['poses']))
            }
        )
        
        # Save observations
        obs_path = demo_path / 'observations'
        obs_path.mkdir(exist_ok=True)
        
        for i, obs in enumerate(demo_data['observations']):
            frame_path = obs_path / f'frame_{i:06d}'
            frame_path.mkdir(exist_ok=True)
            
            for camera_id, camera_data in obs.items():
                np.save(frame_path / f'{camera_id}_color.npy', camera_data['color'])
                np.save(frame_path / f'{camera_id}_depth.npy', camera_data['depth'])
                np.save(frame_path / f'{camera_id}_point_cloud.npy', camera_data['point_cloud'])
                np.save(frame_path / f'{camera_id}_camera_pose.npy', camera_data['camera_pose'])
                np.save(frame_path / f'{camera_id}_camera_intrinsics.npy', camera_data['camera_intrinsics'])
        
        # Save metadata
        np.save(
            demo_path / 'metadata.npy',
            {
                'demo_id': demo_data['demo_id'],
                'object_transforms': demo_data['object_transforms'],
                'waypoints': demo_data['waypoints'],
                'gripper_states': demo_data['gripper_states'],
                'touched_objects': demo_data.get('touched_objects', [None] * len(demo_data['waypoints']))
            }
        ) 