import os
import numpy as np
import pyrender
import trimesh
import rerun as rr
from tqdm import tqdm
from copy import deepcopy
from uuid import uuid4
import gc
import tracemalloc

from agent.policy import PosePolicy, JointPolicy, AbsoluteWaypoint, ObjectCentricWaypoint
from sim.environment import Environment
from agent.robot import RobotState
from sim.camera import Camera
from sim.object import Object
from sim.scene import default_scene
from sim.renderer import Renderer
from object.shapenet import ShapeNet, SHAPENET_DIRECTORY
from agent.widowx import widowx_controller, widowx_renderer
from agent.humanoid import humanoid_controller, humanoid_renderer
from agent.smplh import smplh_controller, smplh_renderer
from agent.unvisualized import unvisualized_controller, unvisualized_renderer
import utils.spatial as spatial_utils
import utils.trimesh as trimesh_utils

def make_humanoid(scene: pyrender.Scene):
    controller = humanoid_controller()
    renderer = humanoid_renderer(scene)
    transform = np.eye(4)
    transform[:3, 3] = np.array([0, 0, -1.15])
    eef_forward_vector = np.array([0, 1, 0])
    return controller, renderer, transform, eef_forward_vector

def make_smplh(scene: pyrender.Scene):
    controller = smplh_controller()
    renderer = smplh_renderer(scene)
    rotation_quat = trimesh.transformations.quaternion_multiply(
        trimesh.transformations.quaternion_about_axis(3 * np.pi/2, [0, 0, 1]),
        trimesh.transformations.quaternion_about_axis(np.pi/2, [1, 0, 0])
    )
    transform = trimesh.transformations.quaternion_matrix(rotation_quat)
    eef_forward_vector = np.array([0, 1, 0])
    return controller, renderer, transform, eef_forward_vector

def make_widowx(scene: pyrender.Scene):
    controller = widowx_controller()
    renderer = widowx_renderer(scene)
    rotation_quat = trimesh.transformations.quaternion_about_axis(np.pi, [0, 0, 1])
    transform = trimesh.transformations.quaternion_matrix(rotation_quat)
    eef_forward_vector = np.array([-1, 0, 0])
    return controller, renderer, transform, eef_forward_vector

def make_unvisualized(scene: pyrender):
    controller = unvisualized_controller()
    renderer = unvisualized_renderer()
    transform = np.eye(4)
    eef_forward_vector = np.array([0, 1, 0])
    return controller, renderer, transform, eef_forward_vector
    
if __name__ == "__main__":
    num_examples = 10000
    num_cameras = 1
    num_objects = 2
    # num_humanoid_demos = 0
    # num_widowx_demos = 1
    num_demo_episodes = 1
    num_execution_episodes = 1
    num_arms = 1
    shapenet = ShapeNet(SHAPENET_DIRECTORY)

    # rr.init("Rigid Manipulation Demo", spawn=True)
    dataset_frame_id = 0
    
    # Create dataset directory if it doesn't exist
    dataset_dir = "dataset_rrd"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")
        starting_example = 0
    else:
        print(f"Directory already exists: {dataset_dir}")
        # overwrite the last 3 examples
        starting_example = max(0, len(os.listdir(dataset_dir)) - 3)

    # tracemalloc.start(25)
    # prev_snapshot = None

    for i, example in enumerate(tqdm(range(starting_example, num_examples), desc="Examples", position=0, leave=True)):        
        if i > 400:
            break
        recording = rr.RecordingStream("Rigid Manipulation Demo", recording_id=uuid4())
        recording.save(f"{dataset_dir}/dataset_{example}.rrd")
        object_meshes = [shapenet.get_random_mesh().apply_scale(0.375) for _ in range(num_objects)]
        object_point_transforms = [trimesh_utils.object_point_and_normal(obj) for obj in object_meshes]
        camera_states = [Camera() for _ in range(num_cameras)]
        recording.set_time("meta_episode_number", sequence=example)

        for demo in range(num_demo_episodes + num_execution_episodes):
            recording.set_time("episode_number", sequence=demo)
            meta_episode_frame_number = 0
            arm_transforms = {}

            default_forward = np.array([1, 0, 0])
            default_up = np.array([0, 0, 1])
            desired_up = np.array([0, 0, 1])

            robot_states = {}
            waypoints = []

            # Create arm controllers for initialization
            scene = default_scene()
            if demo < num_demo_episodes:
                # arm_controllers = {arm_id: make_humanoid(scene) for arm_id in range(num_arms)}
                # arm_controllers = {arm_id: make_smplh(scene) for arm_id in range(num_arms)}
                arm_controllers = {arm_id: make_unvisualized(scene) for arm_id in range(num_arms)}
            else:
                # arm_controllers = {arm_id: make_widowx(scene) for arm_id in range(num_arms)}
                arm_controllers = {arm_id: make_unvisualized(scene) for arm_id in range(num_arms)}

            object_states = {i: Object(bounding_box_radius=0.1) for i in range(num_objects)}
            for arm_id in range(num_arms):                
                controller, renderer, arm_transform, eef_forward_vector = arm_controllers[arm_id]
                
                random_look_at_translation = spatial_utils.spherical_to_cartesian(
                    *spatial_utils.random_spherical_coordinates(min_dist=-0.25, max_dist=-0.35, randomize_elevation=False)
                )
                random_look_at_forward = random_look_at_translation / np.linalg.norm(random_look_at_translation)
                random_look_at_rotation = spatial_utils.look_at_rotation(default_forward.copy(), random_look_at_forward.copy(), default_up.copy(), desired_up.copy())
                random_look_at = np.eye(4)
                random_look_at[:3, :3] = random_look_at_rotation
                random_look_at[:3, 3] = random_look_at_translation
                arm_transform = random_look_at @ arm_transform
                
                initial_eef_pose = arm_transform @ controller.pose
                object_point, object_face_normal = object_point_transforms[0]
                object_point_transform = trimesh.geometry.align_vectors(eef_forward_vector, object_face_normal)
                object_point_transform[:3, 3] = object_point

                robot_states[arm_id] = RobotState(arm_transform)

                waypoints.append((arm_id, AbsoluteWaypoint(object_id=None, pose=initial_eef_pose)))
                waypoints.append((arm_id, ObjectCentricWaypoint(object_id=arm_id, pose=object_point_transform)))
                waypoints.append((arm_id, AbsoluteWaypoint(object_id=arm_id, pose=initial_eef_pose)))

            env = Environment(camera_states=camera_states, object_states=deepcopy(object_states), robot_states=robot_states, finished=False)
            num_steps = 10
            # renderer = Renderer(scene, object_meshes, {arm_id: renderer for arm_id, (_, renderer, _, _) in arm_controllers.items()}, num_cameras, image_width=32, image_height=32)
            renderer = Renderer(scene, object_meshes, {arm_id: renderer for arm_id, (_, renderer, _, _) in arm_controllers.items()}, num_cameras)
            policy = PosePolicy({arm_id: controller for arm_id, (controller, _, _, _) in arm_controllers.items()}, waypoints, env, num_steps=num_steps)
            steps_per_episode = max(len([waypoint for waypoint in waypoints if waypoint[0] == arm_id]) for arm_id in range(num_arms)) * num_steps

            for i in tqdm(range(steps_per_episode), desc=f"Example {example}, Episode {demo}", position=1, leave=False):
                recording.set_time("frame_id", sequence=dataset_frame_id) # globally unique frame id
                dataset_frame_id += 1
                recording.set_time("meta_episode_frame_number", sequence=meta_episode_frame_number)
                meta_episode_frame_number += 1
                recording.set_time("episode_frame_number", sequence=i) # frame number within episode
                action = policy(env)
                env = env(action)
                observations = renderer(env)
                for arm_id, robot_state in env.robot_states.items():
                    recording.log(
                        f"world/arm_{arm_id}/base_pose",
                        rr.Transform3D(
                            mat3x3=robot_state.arm_pose[:3, :3],
                            translation=robot_state.arm_pose[:3, 3]
                        ),
                    )
                    recording.log(
                        f"world/arm_{arm_id}/eef_pose",
                        rr.Transform3D(
                            mat3x3=robot_state.gripper_pose[:3, :3],
                            translation=robot_state.gripper_pose[:3, 3]
                        ),
                    )
                    recording.log(f"world/arm_{arm_id}/object_id", rr.Scalars(robot_state.grasped_object_id))
                    # TODO: this is hacky and not great but tensor is not supported in dataframe
                    recording.log(f"world/arm_{arm_id}/joint_angle", rr.Scalars(robot_state.joint_angle.astype(np.float32)))
                for camera_id, camera_data in enumerate(observations):
                    recording.log(
                        f"world/camera_{camera_id}",
                        rr.Pinhole(
                            image_from_camera=camera_data['camera_intrinsics'],
                            width=camera_data['color'].shape[1],
                            height=camera_data['color'].shape[0],
                            camera_xyz=rr.ViewCoordinates.RUB,
                        ),
                    )
                    recording.log(f"world/camera_{camera_id}", rr.Transform3D(
                        mat3x3=camera_data['camera_pose'][:3, :3],
                        translation=camera_data['camera_pose'][:3, 3],
                    ))
                    recording.log(f"world/camera_{camera_id}/color", rr.Image(camera_data['color']))
                    recording.log(f"world/camera_{camera_id}/depth", rr.DepthImage(camera_data['depth']))
                    recording.log(f"world/camera_{camera_id}/mask", rr.Image(camera_data['mask'], color_model="L"))
                    recording.log(
                        f"world/camera_{camera_id}/seg",
                        rr.SegmentationImage(camera_data['seg']),
                        rr.AnnotationContext([
                            (0, "background", (0, 0, 0)), 
                            (1, "arm", (255, 0, 0)), 
                            (2, "object", (0, 255, 0))
                        ])
                    )
        del recording
        gc.collect()
        
        # # Take a snapshot and compare with previous snapshot
        # current_snapshot = tracemalloc.take_snapshot()
        # if prev_snapshot:
        #     top_stats = current_snapshot.compare_to(prev_snapshot, 'traceback')
        #     print("\nMemory differences since last example:")
        #     for stat in top_stats[:10]:  # top 10 memory differences
        #         print(f"{stat.size_diff / 1024:.1f} KiB: ", end="")
        #         print("\n".join(stat.traceback.format()))
        #         print()
        # else:
        #     # For the first iteration, just show top allocations
        #     top_stats = current_snapshot.statistics('traceback')
        #     print("\nTop memory allocations:")
        #     for stat in top_stats[:10]:
        #         print(f"{stat.size / 1024:.1f} KiB: ", end="")
        #         print("\n".join(stat.traceback.format()))
        #         print()
                
        # prev_snapshot = current_snapshot