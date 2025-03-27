import numpy as np
import mujoco
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from robot_descriptions import widow_mj_description
from dataclasses import dataclass
from typing import Dict, Optional, List

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

@dataclass
class ArmController:
    """
    Controller for a WidowX arm.
    Handles both inverse kinematics and gripper control.
    """
    model: mujoco.MjModel
    data: mujoco.MjData
    eef_id: int
    gripper_joint_ids: List[int]
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, eef_body_name: str, gripper_joint_names: List[str]):
        self.model = model
        self.data = data
        self.eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
        self.gripper_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in gripper_joint_names]
    def __call__(self, target_pose: np.ndarray, open_amount: float, hint: Optional[np.ndarray] = None):
        root_to_gripper_translation = np.mean([self.data.xpos[self.model.jnt_bodyid[child_joint_id]] - self.data.xpos[self.eef_id] for child_joint_id in self.gripper_joint_ids], axis=0)
        root_to_gripper_transform = np.eye(4)
        root_to_gripper_transform[:3, 3] = root_to_gripper_translation
        gripper_to_root_transform = np.linalg.inv(root_to_gripper_transform)
        target_pose = gripper_to_root_transform @ target_pose
        if hint is None:
            hint = self.data.qpos
        self.data.qpos[:self.model.nq] = hint
        mujoco.mj_forward(self.model, self.data)

        max_iterations = 10000
        tolerance = 1e-4
        learning_rate = 0.1

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)  # Compute forward kinematics
            
            # Get current end-effector pose as a homogeneous matrix
            current_translation = self.data.xpos[self.eef_id]
            current_rotation = Rotation.from_quat(self.data.xquat[self.eef_id], scalar_first=True).as_matrix()
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
                break

            # Compute Jacobian for translation and rotation
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.eef_id)
            jac = np.concatenate([jacp, jacr], axis=0)

            # Compute change in joint angles using the Jacobian transpose method
            dq = learning_rate * jac.T @ error

            # Apply joint updates
            self.data.qpos[:self.model.nq] += dq
            mujoco.mj_forward(self.model, self.data)  # Update forward kinematics
        
        left_closed = 0.015
        left_open = 0.037
        right_closed = -0.015
        right_open = -0.037
        
        for joint_id in self.gripper_joint_ids:
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if "left" in joint_name.lower():
                self.data.qpos[joint_id] = left_closed + open_amount * (left_open - left_closed)
            elif "right" in joint_name.lower():
                self.data.qpos[joint_id] = right_closed + open_amount * (right_open - right_closed)
        mujoco.mj_forward(self.model, self.data)
        
        arm_pose = np.eye(4)
        arm_pose[:3, :3] = Rotation.from_quat(self.data.xquat[self.eef_id], scalar_first=True).as_matrix()
        arm_pose[:3, 3] = self.data.xpos[self.eef_id]

        return self.data.qpos, arm_pose

def widowx_controller():
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmController(model, data, "wx250s/gripper_link", ["right_finger", "left_finger"])

@dataclass
class ArmRenderer:
    """
    Renders a WidowX arm.
    """
    model: mujoco.MjModel
    data: mujoco.MjData
    body_nodes: Dict[int, pyrender.Node]
    eef_id: int

    def __init__(self, scene: pyrender.Scene, model: mujoco.MjModel, data: mujoco.MjData, eef_body_name: str):
        self.model = model
        self.data = data
        self.eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
        self.body_nodes = body_nodes_from_model(model, widow_mj_description.PACKAGE_PATH + "/assets")
        [scene.add_node(body_node) for body_node in self.body_nodes.values()]
    def __call__(self, matrix_pose: np.ndarray, qpos: np.ndarray):
        self.data.qpos[:self.model.nq] = qpos
        mujoco.mj_forward(self.model, self.data)
        for body_id, body_node in self.body_nodes.items():
            body_transform = np.eye(4)
            body_transform[:3, :3] = Rotation.from_quat(self.data.xquat[body_id], scalar_first=True).as_matrix()
            body_transform[:3, 3] = self.data.xpos[body_id]
            body_node.matrix = matrix_pose @ body_transform

def widowx_renderer(scene: pyrender.Scene):
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmRenderer(scene, model, data, "wx250s/gripper_link")