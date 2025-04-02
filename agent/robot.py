import numpy as np
import mujoco
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class RobotState:
    gripper_pose: np.ndarray
    arm_pose: np.ndarray
    joint_angle: np.ndarray
    grasped_object_id: Optional[int] = None

    def __init__(
            self, 
            arm_pose: np.ndarray, 
            joint_angle: Optional[np.ndarray] = None,
            gripper_pose: Optional[np.ndarray] = None,
            grasped_object_id: Optional[int] = None
        ):
        self.gripper_pose = gripper_pose if gripper_pose is not None else np.eye(4)
        self.arm_pose = arm_pose
        self.joint_angle = joint_angle if joint_angle is not None else np.array([])
        self.grasped_object_id = grasped_object_id

def body_nodes_from_model(model: mujoco.MjModel, asset_path: str, mesh_extension: str):
    body_to_geom_nodes = {}

    for geom_id in range(model.ngeom):
        geom = model.geom(geom_id)
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = geom.dataid[0]
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            mesh_path = f"{asset_path}/{mesh_name}.{mesh_extension}"
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
        elif geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            capsule_radius = geom.size[0]
            capsule_height = geom.size[1] * 2
            capsule = trimesh.creation.capsule(radius=capsule_radius, height=capsule_height)
            pyrender_mesh = pyrender.Mesh.from_trimesh(capsule)
            mesh_node = pyrender.Node(mesh=pyrender_mesh)
        elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            sphere_radius = geom.size[0]
            sphere = trimesh.creation.icosphere(radius=sphere_radius)
            pyrender_mesh = pyrender.Mesh.from_trimesh(sphere)
            mesh_node = pyrender.Node(mesh=pyrender_mesh)
        elif geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            box_size = geom.size
            box = trimesh.creation.box(extents=2 * box_size)
            pyrender_mesh = pyrender.Mesh.from_trimesh(box)
            mesh_node = pyrender.Node(mesh=pyrender_mesh)
        else:
            continue

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
    Handles both inverse kinematics and gripper control.
    """
    model: mujoco.MjModel
    data: mujoco.MjData
    eef_id: int
    gripper_joint_ids: Optional[List[int]]
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, eef_body_name: str, gripper_joint_names: Optional[List[str]] = None):
        self.model = model
        self.data = data
        self.eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
        if gripper_joint_names:
            self.gripper_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in gripper_joint_names]
        else:
            self.gripper_joint_ids = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def pose(self) -> np.ndarray:
        """
        Returns the current end-effector pose as a 4x4 homogeneous transformation matrix.
        """
        current_translation = self.data.xpos[self.eef_id]
        current_rotation = Rotation.from_quat(self.data.xquat[self.eef_id], scalar_first=True).as_matrix()
        current_pose = np.eye(4)
        current_pose[:3, :3] = current_rotation
        current_pose[:3, 3] = current_translation
        if self.gripper_joint_ids:
            root_to_gripper_translation = np.mean([self.data.xpos[self.model.jnt_bodyid[child_joint_id]] - self.data.xpos[self.eef_id] for child_joint_id in self.gripper_joint_ids], axis=0)
            root_to_gripper_transform = np.eye(4)
            root_to_gripper_transform[:3, 3] = root_to_gripper_translation
            current_pose = root_to_gripper_transform @ current_pose
        return current_pose
    
    def __call__(self, target_pose: np.ndarray, open_amount: float, hint: Optional[np.ndarray] = None):
        if self.gripper_joint_ids:
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
        
        arm_pose = np.eye(4)
        arm_pose[:3, :3] = Rotation.from_quat(self.data.xquat[self.eef_id], scalar_first=True).as_matrix()
        arm_pose[:3, 3] = self.data.xpos[self.eef_id]

        return self.data.qpos, arm_pose
    

@dataclass
class ArmRenderer:
    """
    Renders a WidowX arm.
    """
    model: mujoco.MjModel
    data: mujoco.MjData
    body_nodes: Dict[int, pyrender.Node]
    eef_id: int

    def __init__(self, scene: pyrender.Scene, model: mujoco.MjModel, data: mujoco.MjData, eef_body_name: str, asset_path: str, mesh_extension: str):
        self.model = model
        self.data = data
        self.eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
        self.body_nodes = body_nodes_from_model(model, asset_path, mesh_extension)
        for body_node in self.body_nodes.values():
            scene.add_node(body_node)
    def __call__(self, matrix_pose: np.ndarray, qpos: Optional[np.ndarray] = None):
        if qpos is not None:
            self.data.qpos[:self.model.nq] = qpos
        mujoco.mj_forward(self.model, self.data)
        for body_id, body_node in self.body_nodes.items():
            body_transform = np.eye(4)
            body_transform[:3, :3] = Rotation.from_quat(self.data.xquat[body_id], scalar_first=True).as_matrix()
            body_transform[:3, 3] = self.data.xpos[body_id]
            body_node.matrix = matrix_pose @ body_transform
