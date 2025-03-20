from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pyrender
import trimesh
import rerun as rr
import spatial as spatial_utils
import trajectory as trajectory_utils
from widowx_gripper import create_gripper_node
from tqdm import tqdm

@dataclass
class GraspTarget:
    object_id: int
    start_pose: np.ndarray
    grasp_pose: np.ndarray
    end_pose: np.ndarray

@dataclass
class RobotState:
    gripper_pose: np.ndarray
    grasped_object_id: Optional[int] = None

@dataclass
class EnvironmentState:
    camera_poses: List[np.ndarray]
    object_poses: Dict[int, np.ndarray]
    finished: bool
    policy_state: RobotState

    def __init__(self, num_objects: int, num_cameras: int, bounding_box_size: float = 0.6):
        # Object constants
        object_poses = {
            i: spatial_utils.translate_pose(
                spatial_utils.random_rotation(random_z=True, random_y=False, random_x=False),
                np.array([
                    np.random.uniform(-bounding_box_size / 2, bounding_box_size / 2),
                    np.random.uniform(-bounding_box_size / 2, bounding_box_size / 2),
                    0
                ])
            ) for i in range(num_objects)
        }
        
        # Camera constants
        camera_poses = [
            spatial_utils.look_at_transform(
                spatial_utils.spherical_to_cartesian(
                    *spatial_utils.random_spherical_coordinates()
                )
            )
            for _ in range(num_cameras)
        ]
        
        self.camera_poses = camera_poses
        self.object_poses = object_poses
        self.finished = False
        self.policy_state = RobotState(
            gripper_pose=np.eye(4),
            grasped_object_id=None
        )

def default_scene(add_table: bool = False) -> pyrender.Scene:
    scene = pyrender.Scene()
    if add_table:
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
        scene.add_node(table_node)

    # Add directional lights from different angles for even illumination
    direc_light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(np.pi/4), -np.sin(np.pi/4), 0.0],
        [0.0, np.sin(np.pi/4), np.cos(np.pi/4), 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(direc_light1, pose=light_pose1)
    
    # Add second directional light from opposite angle
    direc_light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    light_pose2 = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(-np.pi/4), -np.sin(-np.pi/4), 0.0],
        [0.0, np.sin(-np.pi/4), np.cos(-np.pi/4), 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(direc_light2, pose=light_pose2)
    
    # Add point light above the scene
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    light_pose3 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(point_light, pose=light_pose3)
    return scene

@dataclass
class Renderer:
    renderer: pyrender.OffscreenRenderer
    scene: pyrender.Scene
    camera_intrinsics: List[np.ndarray]
    camera_nodes: List[pyrender.Node]
    object_nodes: Dict[int, pyrender.Node]
    
    def __init__(self, scene: pyrender.Scene, object_meshes: List[trimesh.Trimesh], num_cameras: int, image_width: int = 640, image_height: int = 480):
        num_objects = len(object_meshes)
        
        # Camera intrinsics
        camera_intrinsics = [
            np.array([
                [image_width / (2 * np.tan(np.pi / 8)), 0, image_width / 2],
                [0, image_height / (2 * np.tan(np.pi / 8)), image_height / 2],
                [0, 0, 1]
            ])
            for _ in range(num_cameras)
        ]

        camera_nodes = [
            pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=image_width / image_height), matrix=np.eye(4))
            for _ in range(num_cameras)
        ]
        object_nodes = {i: pyrender.Node(mesh=pyrender.Mesh.from_trimesh(object_meshes[i]), matrix=np.eye(4)) for i in range(num_objects)}
        [scene.add_node(camera_node) for camera_node in camera_nodes]
        [scene.add_node(object_node) for object_node in object_nodes.values()]

        gripper_node = create_gripper_node()
        rotation_y = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        rotation_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        transform = trimesh.transformations.concatenate_matrices(rotation_y, rotation_x)
        gripper_node.matrix = transform
        gripper_node = pyrender.Node(children=[gripper_node])
        scene.add_node(gripper_node)
        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

        self.renderer = renderer
        self.scene = scene
        self.camera_intrinsics = camera_intrinsics
        self.gripper_node = gripper_node
        self.camera_nodes = camera_nodes
        self.object_nodes = object_nodes
    def __call__(self, state: EnvironmentState):
        for obj_id, obj_pose in state.object_poses.items():
            self.object_nodes[obj_id].matrix = obj_pose
        self.gripper_node.matrix = state.policy_state.gripper_pose
        observations = []
        for cam_idx, camera_node in enumerate(self.camera_nodes):
            camera_node.matrix = state.camera_poses[cam_idx]
            self.scene.main_camera_node = camera_node
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            color, depth = self.renderer.render(self.scene, flags=flags)
            frame_observations = {
                'color': color,
                'depth': depth,
                'camera_intrinsics': self.camera_intrinsics[cam_idx],
                'camera_pose': camera_node.matrix
            }
            observations.append(frame_observations)
        return observations

class Environment:
    def __init__(self, grasps: List[GraspTarget]):
        self.grasps = grasps
    def __call__(self, state: EnvironmentState, action: RobotState) -> EnvironmentState:
        new_state = deepcopy(state)
        if state.policy_state.grasped_object_id is not None:
            gripper_delta = action.gripper_pose @ np.linalg.inv(state.policy_state.gripper_pose)
            new_state.object_poses[state.policy_state.grasped_object_id] = gripper_delta @ state.object_poses[state.policy_state.grasped_object_id]        
        new_state.policy_state.gripper_pose = action.gripper_pose
        new_state.policy_state.grasped_object_id = action.grasped_object_id
        return new_state
    
class Policy:
    def __init__(self, grasps: List[GraspTarget], init_state: EnvironmentState):
        waypoints = []
        object_ids = []
        for i in range(len(grasps)):
            waypoints.append(grasps[i].start_pose.copy())
            waypoints.append(init_state.object_poses[grasps[i].object_id] @ grasps[i].grasp_pose.copy())
            waypoints.append(grasps[i].end_pose.copy())
            object_ids.append(None)
            object_ids.append(grasps[i].object_id)
            object_ids.append(None)
        self.poses, self.object_ids = trajectory_utils.linear_interpolation(waypoints, object_ids)
    def __call__(self, state: EnvironmentState) -> RobotState:
        current_pose = state.policy_state.gripper_pose
        current_grasped_object_id = state.policy_state.grasped_object_id

        if len(self.poses) > 0:
            next_pose = self.poses.pop(0).copy()
            next_object_id = self.object_ids.pop(0)
        else:
            next_pose = current_pose.copy()
            next_object_id = current_grasped_object_id

        next_policy_state = RobotState(
            gripper_pose=next_pose,
            grasped_object_id=next_object_id
        )

        return next_policy_state
    
if __name__ == "__main__":
    num_cameras = 2
    num_objects = 4
    env_state = EnvironmentState(num_objects=num_objects, num_cameras=num_cameras)
    scene = default_scene()
    object_meshes = [trimesh.creation.box(extents=[0.08, 0.08, 0.08]) for _ in range(num_objects)]
    for obj in object_meshes:
        color = np.random.randint(0, 256, size=4)
        color[3] = 255
        obj.visual.vertex_colors = color
    renderer = Renderer(scene, object_meshes, num_cameras=num_cameras)
    grasp_start = spatial_utils.translate_pose(np.eye(4), np.array([0, 0, 0.5]))
    grasp_pose = spatial_utils.random_rotation()
    grasp_end = grasp_start.copy()
    grasps = [
        GraspTarget(object_id=0, start_pose=grasp_start.copy(), grasp_pose=grasp_pose.copy(), end_pose=grasp_end.copy()),
        GraspTarget(object_id=1, start_pose=grasp_start.copy(), grasp_pose=grasp_pose.copy(), end_pose=grasp_end.copy())
    ]
    environment = Environment(grasps)
    policy = Policy(grasps, env_state)
    rr.init("Rigid Manipulation", spawn=True)
    for i in tqdm(range(1000)):
        action = policy(env_state)
        env_state = environment(env_state, action)
        observations = renderer(env_state)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
        for camera_id, camera_data in enumerate(observations):
            rr.log(
                f"world/{camera_id}",
                rr.Pinhole(
                    image_from_camera=camera_data['camera_intrinsics'],
                    width=camera_data['color'].shape[1],
                    height=camera_data['color'].shape[0],
                    camera_xyz=rr.ViewCoordinates.RUB,
                ),
            )
            rr.log(f"world/{camera_id}/color", rr.Image(camera_data['color']))
            rr.log(f"world/{camera_id}/depth", rr.DepthImage(camera_data['depth']))
            rr.log(f"world/{camera_id}", rr.Transform3D(
                mat3x3=camera_data['camera_pose'][:3, :3],
                translation=camera_data['camera_pose'][:3, 3],
            ))