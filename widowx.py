import numpy as np
import mujoco
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from robot_descriptions import widow_mj_description
from dataclasses import dataclass
from typing import Dict, Optional

def body_nodes_from_model(model: mujoco.MjModel, asset_path: str):
    body_to_geom_nodes = {}

    for geom_id in range(model.ngeom):
        geom = model.geom(geom_id)

        if geom.type != mujoco.mjtGeom.mjGEOM_MESH and geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        mesh_id = geom.dataid[0]
        if mesh_id < 0:
            continue

        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        mesh_path = f"{asset_path}/{mesh_name}.stl"
        mesh = trimesh.load(mesh_path)
        scale = model.mesh_scale[mesh_id]
        mesh.apply_scale(scale)
        mesh_pos = model.mesh_pos[mesh_id]
        mesh_quat = model.mesh_quat[mesh_id]
        mesh_rotation = Rotation.from_quat(mesh_quat, scalar_first=True).as_matrix()
        mesh_transform = np.eye(4)
        mesh_transform[:3, :3] = mesh_rotation
        mesh_transform[:3, 3] = mesh_pos
        mesh_transform = np.linalg.inv(mesh_transform)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = pyrender.Node(mesh=pyrender_mesh, matrix=mesh_transform)

        geom_pos = geom.pos
        geom_quat = geom.quat
        geom_rotation = Rotation.from_quat(geom_quat, scalar_first=True).as_matrix()
        geom_transform = np.eye(4)
        geom_transform[:3, :3] = geom_rotation
        geom_transform[:3, 3] = geom_pos
        geom_node = pyrender.Node(matrix=geom_transform, children=[mesh_node])

        parent_body_id = geom.bodyid[0]
        if parent_body_id not in body_to_geom_nodes:
            body_to_geom_nodes[parent_body_id] = []
        body_to_geom_nodes[parent_body_id].append(geom_node)
    body_nodes = {body_id: pyrender.Node(children=body_to_geom_nodes.get(body_id, [])) for body_id in range(model.nbody)}
    return body_nodes

def add_widowx_to_scene(scene: pyrender.Scene):
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_nodes = body_nodes_from_model(model, widow_mj_description.PACKAGE_PATH + "/assets")
    for node in body_nodes.values():
        scene.add_node(node)
    return Arm(model, data, body_nodes)

def ik(model: mujoco.MjModel, data: mujoco.MjData, target_body_id: int, target_pose: np.ndarray):
    max_iterations = 10000
    tolerance = 1e-4
    learning_rate = 0.1

    for i in range(max_iterations):
        mujoco.mj_forward(model, data)  # Compute forward kinematics
        
        # Get current end-effector pose as a homogeneous matrix
        current_translation = data.xpos[target_body_id]
        current_rotation = Rotation.from_quat(data.xquat[target_body_id], scalar_first=True).as_matrix()
        current_pose = np.eye(4)
        current_pose[:3, :3] = current_rotation
        current_pose[:3, 3] = current_translation
        
        # Compute translational error
        translation_error = target_pose[:3, 3] - current_translation
        
        # Compute rotational error via the rotation difference
        R_err = target_pose[:3, :3] @ current_rotation.T
        rotation_error = Rotation.from_matrix(R_err).as_rotvec()  # Axis-angle representation
        
        # Combine errors into a 6-dimensional error vector
        error = np.concatenate([translation_error, rotation_error])
        
        # Check if we are close enough
        if np.linalg.norm(error) < tolerance:
            print(f"Converged in {i} iterations.")
            break

        # Compute Jacobian for translation and rotation
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, target_body_id)
        jac = np.concatenate([jacp, jacr], axis=0)

        # Compute change in joint angles using the Jacobian transpose method
        dq = learning_rate * jac.T @ error

        # Apply joint updates
        data.qpos[:model.nq] += dq
        mujoco.mj_forward(model, data)  # Update forward kinematics
    print(current_pose)

@dataclass
class Arm:
    model: mujoco.MjModel
    data: mujoco.MjData
    body_nodes: Dict[int, pyrender.Node]
    initial_pose: np.ndarray
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_nodes: Dict[int, pyrender.Node]):
        self.model = model
        self.data = data
        self.body_nodes = body_nodes
        self.initial_pose = np.eye(4)
        self.initial_pose[:3, :3] = Rotation.from_quat(self.data.xquat[8], scalar_first=True).as_matrix()
        self.initial_pose[:3, 3] = self.data.xpos[8]
        self.go_to_pose()
    def go_to_pose(self, pose: Optional[np.ndarray] = None):
        if pose is None:
            pose = np.eye(4)
        target_pose = pose @ self.initial_pose
        ik(self.model, self.data, 8, target_pose)
        for body_id, body_node in self.body_nodes.items():
            body_transform = np.eye(4)
            body_transform[:3, :3] = Rotation.from_quat(self.data.xquat[body_id], scalar_first=True).as_matrix()
            body_transform[:3, 3] = self.data.xpos[body_id]
            body_node.matrix = body_transform

scene = pyrender.Scene()
arm = add_widowx_to_scene(scene)
transform = np.eye(4)
transform[:3, 3] = np.array([0.1, 0.0, 0.1])
arm.go_to_pose(transform)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)