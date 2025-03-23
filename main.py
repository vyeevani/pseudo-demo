from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pyrender
import trimesh
import rerun as rr
import spatial as spatial_utils
import trajectory as trajectory_utils
from widowx import widowx_arm, Arm
from tqdm import tqdm
import trimesh_utils as trimesh_utils

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

    def __init__(self, num_objects: int, num_cameras: int, bounding_box_radius: float = 0.1):
        object_poses = {}
        for i in range(num_objects):
            bounding_box_x = np.random.uniform(0, bounding_box_radius)
            bounding_box_y = np.sqrt(bounding_box_radius**2 - bounding_box_x**2) * np.random.choice([-1, 1])
            object_poses[i] = spatial_utils.translate_pose(
                spatial_utils.random_rotation(random_z=True, random_y=False, random_x=False),
                np.array([
                    bounding_box_x,
                    bounding_box_y,
                    0
                ])
            )
        
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
    arm: Arm
    def __init__(self, scene: pyrender.Scene, object_meshes: List[trimesh.Trimesh], num_cameras: int, image_width: int = 480, image_height: int = 480):
        num_objects = len(object_meshes)
        yfov = np.pi/4.0
        fx = image_width / (2 * np.tan(yfov / 2))
        fy = image_height / (2 * np.tan(yfov / 2))
        camera_intrinsics = [
            np.array([
                [fx, 0, image_width / 2],
                [0, fy, image_height / 2],
                [0, 0, 1]
            ])
            for _ in range(num_cameras)
        ]

        camera_nodes = [
            pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=image_width / image_height), matrix=np.eye(4))
            for _ in range(num_cameras)
        ]
        object_nodes = {i: pyrender.Node(mesh=pyrender.Mesh.from_trimesh(object_meshes[i]), matrix=np.eye(4)) for i in range(num_objects)}
        [scene.add_node(camera_node) for camera_node in camera_nodes]
        [scene.add_node(object_node) for object_node in object_nodes.values()]

        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = -0.25
        arm_node, self.arm = widowx_arm(translation_matrix)
        scene.add_node(arm_node)

        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

        self.renderer = renderer
        self.scene = scene
        self.camera_intrinsics = camera_intrinsics
        self.camera_nodes = camera_nodes
        self.object_nodes = object_nodes
    def __call__(self, state: EnvironmentState):
        for obj_id, obj_pose in state.object_poses.items():
            self.object_nodes[obj_id].matrix = obj_pose
        self.arm.go_to_pose(state.policy_state.gripper_pose)
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
        for grasp in grasps:
            waypoints.append(grasp.start_pose.copy())
            object_ids.append(None)
            waypoints.append(init_state.object_poses[grasp.object_id] @ grasp.grasp_pose.copy())
            object_ids.append(grasp.object_id)
            waypoints.append(grasp.end_pose.copy())
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
    num_cameras = 4
    num_objects = 2
    env_state = EnvironmentState(num_objects=num_objects, num_cameras=num_cameras)
    scene = default_scene()
    object_thickness = 0.08
    object_meshes = [trimesh.creation.box(extents=[object_thickness, object_thickness, object_thickness]) for _ in range(num_objects)]
    object_point_transforms = [trimesh_utils.object_point_transform(obj, np.array([-1, 0, 0])) for obj in object_meshes]
    for obj in object_meshes:
        color = np.random.randint(0, 256, size=4)
        color[3] = 255
        obj.visual.vertex_colors = color
    renderer = Renderer(scene, object_meshes, num_cameras=num_cameras)
    grasp_start = spatial_utils.translate_pose(np.eye(4), np.array([0, 0, 0.25]))
    grasp_end = grasp_start.copy()
    grasps = [
        GraspTarget(object_id=0, start_pose=grasp_start.copy(), grasp_pose=object_point_transforms[0], end_pose=grasp_end.copy()),
        GraspTarget(object_id=1, start_pose=grasp_start.copy(), grasp_pose=object_point_transforms[1], end_pose=grasp_end.copy()),
    ]
    environment = Environment(grasps)
    policy = Policy(grasps, env_state)
    rr.init("Rigid Manipulation", spawn=True)
    for i in tqdm(range(250)):
        action = policy(env_state)
        env_state = environment(env_state, action)
        observations = renderer(env_state)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
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