import numpy as np
from dataclasses import dataclass

import spatial as spatial_utils

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