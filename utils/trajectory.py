import numpy as np
import transforms3d as t3d
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Literal, Any

def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Interpolated quaternion
    """
    # Compute the cosine of the angle between the quaternions
    dot = np.dot(q1, q2)
    
    # If the dot product is negative, negate one of the quaternions
    # to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, linearly interpolate
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    # Compute interpolation factors
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

def linear_interpolation(
    waypoints: List[np.ndarray],
    metadata: List[Any],
    num_steps: int = 100,
) -> Tuple[List[np.ndarray], List[Any]]:
    """Linear interpolation between waypoints."""
    interpolated_poses = []
    interpolated_metadata = []
    
    for i in range(len(waypoints) - 1):
        start_pose = waypoints[i]
        end_pose = waypoints[i + 1]
        start_metadata = metadata[i]
        
        for t in np.linspace(0, 1, num_steps):
            # Interpolate position
            pos = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
            
            # Interpolate rotation using SLERP
            start_quat = t3d.quaternions.mat2quat(start_pose[:3, :3])
            end_quat = t3d.quaternions.mat2quat(end_pose[:3, :3])
            interp_quat = slerp(start_quat, end_quat, t)
            rot = t3d.quaternions.quat2mat(interp_quat)
            
            # Combine into transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = pos
            
            interpolated_poses.append(T)
            # Keep metadata constant until next waypoint
            interpolated_metadata.append(start_metadata)
    
    # Add final waypoint
    interpolated_poses.append(waypoints[-1])
    interpolated_metadata.append(metadata[-1])
    
    return interpolated_poses, interpolated_metadata
    
def cubic_interpolation(
    waypoints: List[np.ndarray],
    metadata: List[Any],
    num_steps: int = 100,
) -> Tuple[List[np.ndarray], List[Any]]:
    """Cubic spline interpolation for position, SLERP for rotation."""
    # Extract positions and create parameter space
    positions = np.array([wp[:3, 3] for wp in waypoints])
    t = np.arange(len(waypoints))
    
    # Fit cubic spline to positions
    cs = CubicSpline(t, positions)
    
    t_interp = np.linspace(0, len(waypoints) - 1, num_steps)
    
    interpolated_poses = []
    interpolated_metadata = []
    
    # Interpolate
    for t in t_interp:
        # Get interpolated position
        pos = cs(t)
        
        # Find surrounding waypoints for rotation interpolation
        idx = int(t)
        if idx >= len(waypoints) - 1:
            idx = len(waypoints) - 2
        frac = t - idx
        
        # Interpolate rotation using SLERP
        start_quat = t3d.quaternions.mat2quat(waypoints[idx][:3, :3])
        end_quat = t3d.quaternions.mat2quat(waypoints[idx + 1][:3, :3])
        rot = t3d.quaternions.quat2mat(
            slerp(start_quat, end_quat, frac)
        )
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        
        interpolated_poses.append(T)
        # Interpolate metadata (keep constant between waypoints)
        interpolated_metadata.append(metadata[idx])
    
    return interpolated_poses, interpolated_metadata

def spherical_interpolation(
    waypoints: List[np.ndarray],
    metadata: List[Any],
    num_steps: int = 100,
) -> Tuple[List[np.ndarray], List[Any]]:
    """Interpolation while maintaining constant distance from origin."""
    interpolated_poses = []
    interpolated_metadata = []
    
    for i in range(len(waypoints) - 1):
        start_pose = waypoints[i]
        end_pose = waypoints[i + 1]
        start_metadata = metadata[i]
        
        # Calculate spherical coordinates
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        
        # Convert to spherical coordinates
        start_r = np.linalg.norm(start_pos)
        start_theta = np.arccos(start_pos[2] / start_r)
        start_phi = np.arctan2(start_pos[1], start_pos[0])
        
        end_r = np.linalg.norm(end_pos)
        end_theta = np.arccos(end_pos[2] / end_r)
        end_phi = np.arctan2(end_pos[1], end_pos[0])
        
        for t in np.linspace(0, 1, num_steps):
            # Interpolate spherical coordinates
            r = (1 - t) * start_r + t * end_r
            theta = (1 - t) * start_theta + t * end_theta
            phi = (1 - t) * start_phi + t * end_phi
            
            # Convert back to Cartesian
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            pos = np.array([x, y, z])
            
            # Interpolate rotation using SLERP
            start_quat = t3d.quaternions.mat2quat(start_pose[:3, :3])
            end_quat = t3d.quaternions.mat2quat(end_pose[:3, :3])
            rot = t3d.quaternions.quat2mat(
                slerp(start_quat, end_quat, t)
            )
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = pos
            
            interpolated_poses.append(T)
            interpolated_metadata.append(start_metadata)
    
    # Add final waypoint
    interpolated_poses.append(waypoints[-1])
    interpolated_metadata.append(metadata[-1])
    
    return interpolated_poses, interpolated_metadata 