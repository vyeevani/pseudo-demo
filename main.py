import numpy as np
import pyrender
import trimesh
import rerun as rr
from tqdm import tqdm
from copy import deepcopy

from agent.policy import Policy, GraspTarget
from sim.environment import Environment
from agent.robot import RobotState
from sim.camera import Camera
from sim.object import Object
from sim.scene import default_scene
from sim.renderer import Renderer
from agent.widowx import widowx_controller, widowx_renderer
from agent.humanoid import humanoid_controller, humanoid_renderer

import utils.spatial as spatial_utils
import utils.trimesh as trimesh_utils

def make_humanoid(scene: pyrender.Scene):
    controller = humanoid_controller()
    renderer = humanoid_renderer(scene)
    transform = np.eye(4)
    transform[:3, 3] = np.array([0, 0, -1.15])
    eef_forward_vector = np.array([0, 1, 0])
    return controller, renderer, transform, eef_forward_vector

def make_widowx(scene: pyrender.Scene):
    controller = widowx_controller()
    renderer = widowx_renderer(scene)
    transform = np.eye(4)
    eef_forward_vector = np.array([-1, 0, 0])
    return controller, renderer, transform, eef_forward_vector
    
if __name__ == "__main__":
    num_examples = 1
    num_cameras = 4
    num_objects = 1
    num_humanoid_demos = 1
    num_widowx_demos = 1
    num_arms = 1

    rr.init("Rigid Manipulation Demo", spawn=True)
    # rr.init("Rigid Manipulation Demo")
    # rr.save("dataset.rrd")
    unique_frame_id = 0

    for example in range(num_examples):
        object_meshes = [trimesh.creation.box(extents=[np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)]) for _ in range(num_objects)]
        object_point_transforms = [trimesh_utils.object_point_and_normal(obj) for obj in object_meshes]
        object_states = {i: Object(bounding_box_radius=0.1) for i in range(num_objects)}

        # Initialize camera states once to retain positions between demos
        camera_states = [Camera() for _ in range(num_cameras)]
        rr.set_time_sequence("meta_episode_number", example)
        for demo in range(num_humanoid_demos + num_widowx_demos):
            rr.set_time_sequence("episode_number", demo)
            arm_transforms = {}

            default_forward = np.array([1, 0, 0])
            default_up = np.array([0, 0, 1])
            desired_up = np.array([0, 0, 1])

            robot_states = {}
            grasps = []

            # Create arm controllers for initialization
            scene = default_scene()
            if demo < num_humanoid_demos:
                arm_controllers = {arm_id: make_humanoid(scene) for arm_id in range(num_arms)}
            else:
                arm_controllers = {arm_id: make_widowx(scene) for arm_id in range(num_arms)}

            for arm_id in range(num_arms):                
                controller, renderer, arm_transform, eef_forward_vector = arm_controllers[arm_id]

                arm_translation = spatial_utils.spherical_to_cartesian(
                    *spatial_utils.random_spherical_coordinates(min_dist=-0.25, max_dist=-0.35, randomize_elevation=False)
                )
                desired_forward = -arm_translation / np.linalg.norm(arm_translation)
                arm_rotation = spatial_utils.look_at_rotation(default_forward.copy(), desired_forward.copy(), default_up.copy(), desired_up.copy())
                arm_transform[:3, :3] = arm_rotation
                arm_transform[:3, 3] += arm_translation

                robot_states[arm_id] = RobotState(arm_transform)
                
                # Arm grasps
                grasp_start_transform = np.eye(4)
                grasp_start = arm_transform @ controller.pose
                grasp_end = grasp_start.copy()

                object_point, object_face_normal = object_point_transforms[0]
                object_point_transform = trimesh.geometry.align_vectors(eef_forward_vector, object_face_normal)
                object_point_transform[:3, 3] = object_point

                # Create grasp for the arm
                grasps.append(
                    GraspTarget(
                        object_id=arm_id, 
                        arm_id=arm_id, 
                        start_pose=grasp_start.copy(), 
                        grasp_pose=object_point_transform, 
                        end_pose=grasp_end.copy()
                    )
                )

            # object_states = {i: Object(bounding_box_radius=0.1) for i in range(num_objects)}
            object_states = deepcopy(object_states)
            env = Environment(camera_states=camera_states, object_states=object_states, robot_states=robot_states, finished=False)
            num_steps = 25
            policy = Policy({arm_id: controller for arm_id, (controller, _, _, _) in arm_controllers.items()}, grasps, env, num_steps=num_steps)
            renderer = Renderer(scene, object_meshes, {arm_id: renderer for arm_id, (_, renderer, _, _) in arm_controllers.items()}, num_cameras)
            steps_per_episode = max(len([grasp for grasp in grasps if grasp.arm_id == arm_id]) for arm_id in range(num_arms)) * 3 * num_steps
            steps_per_episode = 100

            for i in tqdm(range(steps_per_episode)):
                rr.set_time_sequence("frame_id", unique_frame_id) # globally unique frame id
                unique_frame_id += 1
                rr.set_time_sequence("frame_number", i) # frame number within episode
                action = policy(env)
                env = env(action)
                observations = renderer(env)
                for arm_id, robot_state in env.robot_states.items():
                    rr.log(
                        f"world/arm_{arm_id}/pose",
                        rr.Transform3D(
                            mat3x3=robot_state.gripper_pose[:3, :3],
                            translation=robot_state.gripper_pose[:3, 3],
                        ),
                    )
                    rr.log(f"world/arm_{arm_id}/object_id", rr.Scalar(robot_state.grasped_object_id))
                    # TODO: this is hacky and not great but tensor is not supported in dataframe
                    rr.log(f"world/arm_{arm_id}/joint_angle", rr.Scalar(robot_state.joint_angle.astype(np.float32)))
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
                    rr.log(f"world/{camera_id}/mask", rr.Image(camera_data['mask'], color_model="L"))