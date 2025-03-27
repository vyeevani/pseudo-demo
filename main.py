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
    arm_id: int
    start_pose: np.ndarray
    grasp_pose: np.ndarray
    end_pose: np.ndarray

@dataclass
class RobotState:
    gripper_pose: np.ndarray
    grasped_object_id: Optional[int] = None

    def __init__(self):
        self.gripper_pose = np.eye(4)
        self.grasped_object_id = None

@dataclass
class ObjectState:
    pose: np.ndarray

    def __init__(self, bounding_box_radius: float = 0.1):
        bounding_box_x = np.random.uniform(0, bounding_box_radius)
        bounding_box_y = np.sqrt(bounding_box_radius**2 - bounding_box_x**2) * np.random.choice([-1, 1])
        self.pose = spatial_utils.translate_pose(
            spatial_utils.random_rotation(random_z=True, random_y=False, random_x=False),
            np.array([
                bounding_box_x,
                bounding_box_y,
                0
            ])
        )

@dataclass
class CameraState:
    pose: np.ndarray

    def __init__(self):
        default_forward = np.array([0, 0, -1])
        default_up = np.array([0, 1, 0])
        desired_forward = -spatial_utils.spherical_to_cartesian(
            *spatial_utils.random_spherical_coordinates()
        )
        desired_up = np.array([0, 0, 1])
        self.pose = spatial_utils.look_at_rotation(default_forward, desired_forward, default_up, desired_up)
        self.pose = np.hstack((self.pose, -desired_forward.reshape(3, 1)))
        self.pose = np.vstack((self.pose, np.array([0, 0, 0, 1])))

@dataclass
class EnvironmentState:
    camera_states: List[CameraState]
    object_states: Dict[int, ObjectState]
    robot_states: Dict[int, RobotState]
    finished: bool

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
    arms: Dict[int, Arm]
    gripper_speed: float = 0.1  # Speed of gripper movement per step

    def __init__(self, scene: pyrender.Scene, object_meshes: List[trimesh.Trimesh], num_cameras: int, image_width: int = 480, image_height: int = 480, arm_transforms: Dict[int, np.ndarray] = {0: np.eye(4)}):
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

        # Add arms
        self.arms = {}
        for arm_id, arm_transform in arm_transforms.items():
            arm_node, arm = widowx_arm(arm_transform)
            scene.add_node(arm_node)
            self.arms[arm_id] = arm

        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

        self.renderer = renderer
        self.scene = scene
        self.camera_intrinsics = camera_intrinsics
        self.camera_nodes = camera_nodes
        self.object_nodes = object_nodes

    def __call__(self, state: EnvironmentState):
        for obj_id, obj_state in state.object_states.items():
            self.object_nodes[obj_id].matrix = obj_state.pose
        
        # Control all arms based on their poses in policy state
        for arm_id, arm in self.arms.items():
            if arm_id in state.robot_states:
                robot_state = state.robot_states[arm_id]
                target_open = 1.0 if robot_state.grasped_object_id is None else 0.0
                current_open = 1.0 if state.robot_states[arm_id].grasped_object_id is None else 0.0
                
                # Gradually move towards target
                if current_open < target_open:
                    current_open = min(current_open + self.gripper_speed, target_open)
                elif current_open > target_open:
                    current_open = max(current_open - self.gripper_speed, target_open)
                
                arm.go_to_pose(robot_state.gripper_pose, open_amount=current_open)
        
        observations = []
        for cam_idx, camera_state in enumerate(state.camera_states):
            camera_node = self.camera_nodes[cam_idx]
            camera_node.matrix = camera_state.pose
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
    def __call__(self, state: EnvironmentState, action: Dict[int, RobotState]) -> EnvironmentState:
        new_state = deepcopy(state)
        
        # For each arm, update object position if grasping something
        for arm_id, robot_state in action.items():
            if robot_state.grasped_object_id is not None:
                gripper_delta = robot_state.gripper_pose @ np.linalg.inv(state.robot_states[arm_id].gripper_pose)
                new_state.object_states[robot_state.grasped_object_id].pose = gripper_delta @ state.object_states[robot_state.grasped_object_id].pose
        
        new_state.robot_states = action
        return new_state
    
class Policy:
    def __init__(self, grasps: List[GraspTarget], init_state: EnvironmentState, num_steps: int = 25):
        # Create waypoints for each arm separately
        arm_waypoints = {}
        arm_object_ids = {}
        object_states = {obj_id: deepcopy(obj_state) for obj_id, obj_state in init_state.object_states.items()}
        
        # Initialize empty lists for all arms in the state
        for arm_id in init_state.robot_states.keys():
            arm_waypoints[arm_id] = []
            arm_object_ids[arm_id] = []
        
        for grasp in grasps:
            arm_id = grasp.arm_id
            arm_waypoints[arm_id].append(grasp.start_pose.copy())
            arm_object_ids[arm_id].append(None)
            arm_waypoints[arm_id].append(object_states[grasp.object_id].pose @ grasp.grasp_pose.copy())
            arm_object_ids[arm_id].append(grasp.object_id)
            arm_waypoints[arm_id].append(grasp.end_pose.copy())
            arm_object_ids[arm_id].append(None)
            gripper_delta = grasp.end_pose @ np.linalg.inv(object_states[grasp.object_id].pose @ grasp.grasp_pose.copy())
            object_states[grasp.object_id].pose = gripper_delta @ object_states[grasp.object_id].pose
        
        # Generate separate trajectories for each arm
        self.arm_trajectories = {}
        self.arm_object_id_trajectories = {}
        
        for arm_id in arm_waypoints.keys():
            if arm_waypoints[arm_id]:  # If this arm has waypoints
                poses, object_ids = trajectory_utils.linear_interpolation(
                    arm_waypoints[arm_id], 
                    arm_object_ids[arm_id], 
                    num_steps=num_steps
                )
                self.arm_trajectories[arm_id] = poses
                self.arm_object_id_trajectories[arm_id] = object_ids
            else:  # If this arm has no waypoints, create empty lists
                self.arm_trajectories[arm_id] = []
                self.arm_object_id_trajectories[arm_id] = []
                
    def __call__(self, state: EnvironmentState) -> Dict[int, RobotState]:
        # Get all arm IDs from current state
        next_policy_state = {}
        
        # Copy current state
        for arm_id, robot_state in state.robot_states.items():
            next_policy_state[arm_id] = RobotState()
            next_policy_state[arm_id].gripper_pose = robot_state.gripper_pose.copy()
            next_policy_state[arm_id].grasped_object_id = robot_state.grasped_object_id
            
        # Update each arm if it has remaining waypoints
        for arm_id in self.arm_trajectories.keys():
            if self.arm_trajectories[arm_id]:  # If this arm has waypoints left
                next_policy_state[arm_id].gripper_pose = self.arm_trajectories[arm_id].pop(0).copy()
                next_policy_state[arm_id].grasped_object_id = self.arm_object_id_trajectories[arm_id].pop(0)

        return next_policy_state
    
if __name__ == "__main__":
    num_examples = 2
    num_demonstrations = 2
    num_cameras = 4
    num_objects = 4
    arm_ids = [0, 1]

    rr.init("Rigid Manipulation Demo")
    rr.save("dataset.rrd")
    unique_frame_id = 0

    for example in range(num_examples):
        object_meshes = [trimesh.creation.box(extents=[np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)]) for _ in range(num_objects)]
        object_point_transforms = [trimesh_utils.object_point_transform(obj, np.array([-1, 0, 0])) for obj in object_meshes]

        # Initialize camera states once to retain positions between demos
        camera_states = [CameraState() for _ in range(num_cameras)]
        rr.set_time_sequence("meta_episode_number", example)
        for demo in range(num_demonstrations):
            rr.set_time_sequence("episode_number", demo)
            arm_transforms = {}

            default_forward = np.array([1, 0, 0])
            default_up = np.array([0, 0, 1])
            desired_up = np.array([0, 0, 1])

            arm_0_translation = spatial_utils.spherical_to_cartesian(
                *spatial_utils.random_spherical_coordinates(min_dist=-0.35, max_dist=-0.45, randomize_elevation=False)
            )
            desired_forward_0 = -arm_0_translation / np.linalg.norm(arm_0_translation)
            arm_0_rotation = spatial_utils.look_at_rotation(default_forward, desired_forward_0, default_up, desired_up)
            arm_0_transform = np.eye(4)
            arm_0_transform[:3, :3] = arm_0_rotation
            arm_0_transform[:3, 3] = arm_0_translation
            arm_transforms[0] = arm_0_transform

            arm_1_translation = spatial_utils.spherical_to_cartesian(
                *spatial_utils.random_spherical_coordinates(min_dist=-0.35, max_dist=-0.45, randomize_elevation=False)
            )
            desired_forward_1 = -arm_1_translation / np.linalg.norm(arm_1_translation)
            arm_1_rotation = spatial_utils.look_at_rotation(default_forward, desired_forward_1, default_up, desired_up)
            arm_1_transform = np.eye(4)
            arm_1_transform[:3, :3] = arm_1_rotation
            arm_1_transform[:3, 3] = arm_1_translation
            arm_transforms[1] = arm_1_transform
            
            # Arm 0 grasps
            grasp_start_transform0 = np.eye(4)
            grasp_start_transform0[2, 3] = 0.25
            grasp_start_transform0[0, 3] = 0.25
            grasp_start0 = arm_transforms[0] @ grasp_start_transform0
            grasp_end0 = grasp_start0.copy()
            
            # Arm 1 grasps
            grasp_start_transform1 = np.eye(4)
            grasp_start_transform1[2, 3] = 0.25
            grasp_start_transform1[0, 3] = 0.25
            grasp_start1 = arm_transforms[1] @ grasp_start_transform1
            grasp_end1 = grasp_start1.copy()
            
            # Create grasps for both arms
            grasps = [
                GraspTarget(object_id=0, arm_id=0, start_pose=grasp_start0.copy(), grasp_pose=object_point_transforms[0], end_pose=grasp_end0.copy()),
                GraspTarget(object_id=1, arm_id=1, start_pose=grasp_start1.copy(), grasp_pose=object_point_transforms[1], end_pose=grasp_end1.copy()),
            ]

            object_states = {i: ObjectState(bounding_box_radius=0.1) for i in range(num_objects)}
            robot_states = {arm_id: RobotState() for arm_id in arm_ids}
            env_state = EnvironmentState(camera_states=camera_states, object_states=object_states, robot_states=robot_states, finished=False)
            environment = Environment(grasps)
            num_steps = 25
            policy = Policy(grasps, env_state, num_steps=num_steps)
            renderer = Renderer(default_scene(), object_meshes, num_cameras=num_cameras, arm_transforms=arm_transforms)
            steps_per_episode = max(len([grasp for grasp in grasps if grasp.arm_id == arm_id]) for arm_id in arm_ids) * 3 * num_steps
            steps_per_episode = 100

            for i in tqdm(range(steps_per_episode)):
                rr.set_time_sequence("frame_id", unique_frame_id) # globally unique frame id
                unique_frame_id += 1
                rr.set_time_sequence("frame_number", i) # frame number within episode
                action = policy(env_state)
                env_state = environment(env_state, action)
                observations = renderer(env_state)
                for arm_id, robot_state in env_state.robot_states.items():
                    rr.log(
                        f"world/arm_{arm_id}/pose",
                        rr.Transform3D(
                            mat3x3=robot_state.gripper_pose[:3, :3],
                            translation=robot_state.gripper_pose[:3, 3],
                        ),
                    )
                    rr.log(f"world/arm_{arm_id}/object_id", rr.Scalar(robot_state.grasped_object_id))
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
                    rr.log(f"world/{camera_id}", rr.Transform3D(
                        mat3x3=camera_data['camera_pose'][:3, :3],
                        translation=camera_data['camera_pose'][:3, 3],
                    ))
                    rr.log(f"world/{camera_id}/color", rr.Image(camera_data['color']))
                    rr.log(f"world/{camera_id}/depth", rr.DepthImage(camera_data['depth']))