from dataclasses import dataclass

import numpy as np

import utils.spatial as spatial_utils

@dataclass
class Camera:
    pose: np.ndarray

    def __init__(self):
        default_forward = np.array([0, 0, -1]).copy()
        default_up = np.array([0, 1, 0]).copy()
        desired_forward = -spatial_utils.spherical_to_cartesian(
            *spatial_utils.random_spherical_coordinates()
        ).copy()
        desired_up = np.array([0, 0, 1]).copy()
        self.pose = spatial_utils.look_at_rotation(default_forward, desired_forward, default_up, desired_up).copy()
        self.pose = np.hstack((self.pose, -desired_forward.reshape(3, 1))).copy()
        self.pose = np.vstack((self.pose, np.array([0, 0, 0, 1]))).copy()
