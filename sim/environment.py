from dataclasses import dataclass
from typing import List, Dict
from typing_extensions import Self
from copy import deepcopy

import numpy as np

from sim.camera import Camera
from sim.object import Object
from agent.robot import RobotState

@dataclass
class Environment:
    camera_states: List[Camera]
    object_states: Dict[int, Object]
    robot_states: Dict[int, RobotState]
    finished: bool

    def __call__(self, action: Dict[int, RobotState]) -> Self:
        new_state = deepcopy(self)
        
        # For each arm, update object position if grasping something
        for arm_id, robot_state in action.items():
            if robot_state.grasped_object_id is not None:
                gripper_delta = robot_state.gripper_pose @ np.linalg.inv(self.robot_states[arm_id].gripper_pose)
                new_state.object_states[robot_state.grasped_object_id].pose = gripper_delta @ self.object_states[robot_state.grasped_object_id].pose
        
        new_state.robot_states = action
        return new_state
