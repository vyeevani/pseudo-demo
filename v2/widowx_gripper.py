import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation

def create_gripper_node(asset_dir='assets'):
    """
    Create a single pyrender.Node object that renders the entire gripper,
    by loading the gripper meshes with appropriate transforms and attaching them to a parent node.
    
    The gripper comprises:
      - The main gripper mesh ("wx250s_7_gripper.stl")
      - The gripper bar mesh ("wx250s_9_gripper_bar.stl")
      - Two finger meshes ("wx250s_10_gripper_finger.stl") with separate transforms.
    
    Parameters:
        asset_dir (str): Directory where the mesh files are stored.
        
    Returns:
        pyrender.Node: A single parent pyrender.Node object for the entire gripper.
    """
    gripper_mesh = trimesh.load_mesh(f'{asset_dir}/wx250s_7_gripper.stl').apply_scale(0.001)
    gripper_bar_mesh = trimesh.load_mesh(f'{asset_dir}/wx250s_9_gripper_bar.stl').apply_scale(0.001)
    finger_mesh = trimesh.load_mesh(f'{asset_dir}/wx250s_10_gripper_finger.stl').apply_scale(0.001)

    pos_shift = np.array([-0.02, 0, 0])
    quaternion = np.array([1, 0, 0, 1])
    rotation_matrix = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    root_to_motor_transform = np.eye(4)
    root_to_motor_transform[:3, :3] = rotation_matrix
    root_to_motor_transform[:3, 3] = pos_shift
    
    # Setup for root to finger body transformation
    root_to_finger_body = np.eye(4)
    root_to_finger_body[:3, 3] = [0.066, 0, 0]

    # Left finger setup
    left_quaternion = np.array([0, 0, 0, -1])
    left_rotation_matrix = Rotation.from_quat(left_quaternion, scalar_first=True).as_matrix()
    left_finger_body_to_finger_geom = np.eye(4)
    left_finger_body_to_finger_geom[:3, :3] = left_rotation_matrix
    left_finger_body_to_finger_geom[:3, 3] = [0, 0.005 + 0.015, 0]

    root_to_left_finger = root_to_finger_body @ left_finger_body_to_finger_geom

    # Right finger setup
    right_quaternion = np.array([0, 0, 1, 0])
    right_rotation_matrix = Rotation.from_quat(right_quaternion, scalar_first=True).as_matrix()
    right_finger_body_to_finger_geom = np.eye(4)
    right_finger_body_to_finger_geom[:3, :3] = right_rotation_matrix
    right_finger_body_to_finger_geom[:3, 3] = [0, -0.005 - 0.015, 0]

    root_to_right_finger = root_to_finger_body @ right_finger_body_to_finger_geom

    gripper_pyrender_mesh = pyrender.Mesh.from_trimesh(gripper_mesh)
    gripper_bar_pyrender_mesh = pyrender.Mesh.from_trimesh(gripper_bar_mesh)
    left_finger_pyrender_mesh = pyrender.Mesh.from_trimesh(finger_mesh)
    right_finger_pyrender_mesh = pyrender.Mesh.from_trimesh(finger_mesh)

    # Create a parent node for the gripper with all parts as children
    gripper_node = pyrender.Node(children=[
        pyrender.Node(mesh=gripper_pyrender_mesh, matrix=root_to_motor_transform),
        pyrender.Node(mesh=gripper_bar_pyrender_mesh, matrix=root_to_motor_transform),
        pyrender.Node(mesh=left_finger_pyrender_mesh, matrix=root_to_left_finger),
        pyrender.Node(mesh=right_finger_pyrender_mesh, matrix=root_to_right_finger)
    ])

    return gripper_node

# Example usage:
if __name__ == '__main__':
    import pyrender
    import trimesh

    # Create a scene
    scene = pyrender.Scene()

    # Create the gripper node and add it to the scene
    gripper_node = create_gripper_node()
    scene.add_node(gripper_node)
    
    # Set up the viewer to render the scene
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)