import mujoco
import pyrender
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self):
        self.pose = np.eye(4)
    def pose(self):
        return self.pose
    def compute_pose(self, qpos: np.ndarray):
        translation = qpos[:3]
        rotation = qpos[3:]
        self.pose[:3, 3] = translation
        self.pose[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        return self.pose
    def __call__(self, target_pose: np.ndarray, gripper_open_amount: float, hint: Optional[np.ndarray] = None):
        target_translation = target_pose[:3, 3]
        target_rotation = target_pose[:3, :3]
        target_rotation = Rotation.from_matrix(target_rotation).as_quat()
        target_qpos = np.concatenate([target_translation, target_rotation])
        return target_qpos, target_pose
    
class Renderer:
    def __init__(self):
        self.pose = np.eye(4)
        self.body_nodes = {}
    def __call__(self, matrix_pose: np.ndarray, qpos: Optional[np.ndarray] = None):
        if qpos is not None:
            self.pose = qpos
        return self.pose

def unvisualized_controller():
    return Controller()

def unvisualized_renderer():
    return Renderer()

