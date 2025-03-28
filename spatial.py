import numpy as np

import numpy as np

def look_at_rotation(default_forward, desired_forward, default_up, desired_up) -> np.ndarray:
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Zero vector cannot be normalized.")
        return v / norm

    # Normalize forward vectors.
    default_f = normalize(default_forward)
    desired_f = normalize(desired_forward)

    # Compute right vectors (ensure they are perpendicular to forward).
    default_r = normalize(np.cross(default_up, default_f))
    desired_r = normalize(np.cross(desired_up, desired_f))

    # Recompute up vectors to ensure orthogonality.
    default_u = np.cross(default_f, default_r)
    desired_u = np.cross(desired_f, desired_r)

    # Form basis matrices. Each column represents a basis vector: [right, up, forward].
    default_basis = np.stack([default_r, default_u, default_f], axis=1)
    desired_basis = np.stack([desired_r, desired_u, desired_f], axis=1)

    # Calculate the rotation matrix.
    rotation_matrix = desired_basis @ default_basis.T
    return rotation_matrix


def random_spherical_coordinates(
        min_dist: float = 1.0, 
        max_dist: float = 2.0, 
        randomize_azimuth: bool = True, 
        randomize_elevation: bool = True, 
    ) -> np.ndarray:
    """Generate random spherical coordinates."""
    if randomize_azimuth:
        theta = np.random.uniform(0, 2 * np.pi)  # azimuth
    else:
        theta = 0
    if randomize_elevation:
        phi = np.random.uniform(np.pi/6, np.pi/2)  # elevation (avoid too low angles)
    else:
        phi = np.pi/2
    r = np.random.uniform(min_dist, max_dist)  # distance
    return theta, phi, r
def spherical_to_cartesian(theta: float, phi: float, r: float) -> np.ndarray:
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return np.array([x, y, z])

def random_rotation(random_x: bool = True, random_y: bool = True, random_z: bool = True) -> np.ndarray:
    """Generate a random 4x4 pose matrix with random rotations around specified axes."""
    # Determine which axes to apply random rotations
    angles = np.zeros(3)
    if random_x:
        angles[0] = np.random.uniform(0, 2 * np.pi)
    if random_y:
        angles[1] = np.random.uniform(0, 2 * np.pi)
    if random_z:
        angles[2] = np.random.uniform(0, 2 * np.pi)
    
    # Rotation matrices for x, y, z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    
    return T.copy()

def translate_pose(pose: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Translate a pose matrix to a new translation."""
    T = np.eye(4)
    T[:3, :3] = pose[:3, :3]
    T[:3, 3] = translation
    return T