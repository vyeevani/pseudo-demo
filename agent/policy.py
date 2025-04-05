from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from sim.environment import Environment
from agent.robot import ArmController, RobotState
import utils.trajectory as trajectory_utils

@dataclass
class ObjectCentricWaypoint: 
    object_id: int
    pose: np.ndarray

@dataclass
class AbsoluteWaypoint:
    pose: np.ndarray
    object_id: Optional[int] = None

Waypoint = ObjectCentricWaypoint | AbsoluteWaypoint

class WaypointPolicyBase:
    def __init__(self, arm_controllers: Dict[int, ArmController], waypoints: List[Tuple[int, Waypoint]], env: Environment, num_steps: int = 25):
        # Create waypoints for each arm separately
        self.arm_waypoints = {}
        self.arm_object_ids = {}
        self.object_states = {obj_id: deepcopy(obj_state) for obj_id, obj_state in env.object_states.items()}

        # Initialize empty lists for all arms in the state
        for arm_id in env.robot_states.keys():
            self.arm_waypoints[arm_id] = []
            self.arm_object_ids[arm_id] = []

        for arm_id, waypoint in waypoints:
            match waypoint:
                case ObjectCentricWaypoint(object_id, pose):
                    gripper_pose = self.object_states[object_id].pose.copy() @ pose.copy()
                    self.arm_waypoints[arm_id].append(gripper_pose)
                    self.arm_object_ids[arm_id].append(object_id)
                case AbsoluteWaypoint(pose, object_id):
                    gripper_pose = pose.copy()
                    self.arm_waypoints[arm_id].append(gripper_pose)
                    self.arm_object_ids[arm_id].append(object_id)
                    if object_id != None:
                        gripper_delta = gripper_pose.copy() @ np.linalg.inv(self.object_states[object_id].pose.copy())
                        self.object_states[object_id].pose = gripper_delta.copy() @ self.object_states[object_id].pose.copy()

        self.arm_trajectories = {}
        self.arm_object_id_trajectories = {}
        self.arm_joint_angle_trajectories = {}

    def __call__(self, env: Environment) -> Dict[int, RobotState]:
        # Get all arm IDs from current state
        next_policy_state = {}
        
        # Copy current state
        for arm_id, robot_state in env.robot_states.items():
            next_policy_state[arm_id] = RobotState(robot_state.arm_pose.copy(), robot_state.joint_angle.copy() if robot_state.joint_angle is not None else None)
            next_policy_state[arm_id].gripper_pose = robot_state.gripper_pose.copy()
            next_policy_state[arm_id].grasped_object_id = robot_state.grasped_object_id
            
        # Update each arm if it has remaining waypoints
        for arm_id in self.arm_trajectories.keys():
            if self.arm_trajectories[arm_id]:  # If this arm has waypoints left
                next_policy_state[arm_id].gripper_pose = self.arm_trajectories[arm_id].pop(0).copy()
                next_policy_state[arm_id].grasped_object_id = self.arm_object_id_trajectories[arm_id].pop(0)
                next_policy_state[arm_id].joint_angle = self.arm_joint_angle_trajectories[arm_id].pop(0).copy()

        return next_policy_state

class PosePolicy(WaypointPolicyBase):
    def __init__(self, arm_controllers: Dict[int, ArmController], waypoints: List[Tuple[int, Waypoint]], env: Environment, num_steps: int = 25):
        super().__init__(arm_controllers, waypoints, env, num_steps)
        
        for arm_id in self.arm_waypoints.keys():
            if self.arm_waypoints[arm_id]:  # If this arm has waypoints
                poses, object_ids = trajectory_utils.linear_interpolation(
                    self.arm_waypoints[arm_id], 
                    self.arm_object_ids[arm_id], 
                    num_steps=num_steps
                )
                
                # Compute IK for all waypoints upfront and cache
                joint_angle_trajectory = []
                arm_controller = arm_controllers[arm_id]
                previous_joint_angle = None
                
                for pose, obj_id in tqdm(zip(poses, object_ids), total=len(poses), desc=f"Processing arm {arm_id} waypoints"):
                    # Convert to arm frame
                    local_pose = np.linalg.inv(env.robot_states[arm_id].arm_pose) @ pose
                    open_amount = 1.0 if obj_id is None else 0.0
                    # ArmController.__call__ returns (joint_angle, arm_pose)
                    joint_angle, _ = arm_controller(local_pose, open_amount, previous_joint_angle)
                    joint_angle_trajectory.append(joint_angle.copy())
                    previous_joint_angle = joint_angle.copy()
                
                self.arm_trajectories[arm_id] = poses
                self.arm_object_id_trajectories[arm_id] = object_ids
                self.arm_joint_angle_trajectories[arm_id] = joint_angle_trajectory
            else:  # If this arm has no waypoints, create empty lists
                self.arm_trajectories[arm_id] = []
                self.arm_object_id_trajectories[arm_id] = []
                self.arm_joint_angle_trajectories[arm_id] = []

def joint_linear_interpolation(joint_angles_list, metadata_list, num_steps):
    interpolated_joint_angles_list = []
    interpolated_metadata_list = []
    for i in range(len(joint_angles_list) - 1):
        start_angles = joint_angles_list[i]
        end_angles = joint_angles_list[i + 1]
        metadata = metadata_list[i]
        for step in range(num_steps):
            alpha = step / num_steps
            interpolated_angles = (1 - alpha) * start_angles + alpha * end_angles
            interpolated_joint_angles_list.append(interpolated_angles)
            interpolated_metadata_list.append(metadata)
    interpolated_joint_angles_list.append(joint_angles_list[-1])
    interpolated_metadata_list.append(metadata_list[-1])
    return interpolated_joint_angles_list, interpolated_metadata_list

class JointPolicy(WaypointPolicyBase):
    def __init__(self, arm_controllers: Dict[int, ArmController], waypoints: List[Tuple[int, Waypoint]], env: Environment, num_steps: int = 25):
        super().__init__(arm_controllers, waypoints, env, num_steps)
        
        for arm_id in self.arm_waypoints.keys():
            if self.arm_waypoints[arm_id]:
                joint_angle_trajectory = []
                arm_controller = arm_controllers[arm_id]
                for waypoint in self.arm_waypoints[arm_id]:
                    local_pose = np.linalg.inv(env.robot_states[arm_id].arm_pose) @ waypoint
                    open_amount = 1.0
                    joint_angle, _ = arm_controller(local_pose, open_amount)
                    joint_angle_trajectory.append(joint_angle.copy())
                    
                interpolated_joint_angles_list, interpolated_metadata_list = joint_linear_interpolation(
                    joint_angle_trajectory,
                    self.arm_object_ids[arm_id],
                    num_steps
                )
                
                self.arm_trajectories[arm_id] = []
                self.arm_object_id_trajectories[arm_id] = []
                self.arm_joint_angle_trajectories[arm_id] = []
                
                for joint_angle, metadata in zip(interpolated_joint_angles_list, interpolated_metadata_list):
                    arm_pose = env.robot_states[arm_id].arm_pose @ arm_controller.compute_pose(joint_angle)
                    self.arm_trajectories[arm_id].append(arm_pose)
                    self.arm_object_id_trajectories[arm_id].append(metadata)
                    self.arm_joint_angle_trajectories[arm_id].append(joint_angle)
            else:  # If this arm has no waypoints, create empty lists
                self.arm_trajectories[arm_id] = []
                self.arm_object_id_trajectories[arm_id] = []
                self.arm_joint_angle_trajectories[arm_id] = []