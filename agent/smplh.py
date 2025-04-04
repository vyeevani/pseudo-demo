from agent.robot import ArmController, MujocoRenderer
import mujoco
import pyrender
import numpy as np
import os
import smplx
import trimesh
import torch
from typing import Dict

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

def smplh_controller():
    file_path = os.path.join(os.path.dirname(__file__), "smplh", "smplh.xml")
    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    static_body_names = [
        "Pelvis",
        "L_Hip",
        "R_Hip",
        "Torso",
        "L_Knee",
        "R_Knee",
        "Spine",
        "L_Ankle",
        "R_Ankle",
        # "Chest",
        "L_Toe",
        "R_Toe",
        "Neck",
        # "L_Thorax",
        # "R_Thorax",
        "Head",
    ]
    return ArmController(model, data, "R_Wrist", ["R_Pinky3", "R_Thumb3"], static_body_names)

def smplh_pose_from_mujoco(model: mujoco.MjModel, data: mujoco.MjData):
    pose = [0]
    for i, name in enumerate(SMPLH_BONE_ORDER_NAMES):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_ids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body_id]
        for joint_id in joint_ids:
            pose.append(data.qpos[joint_id])
    pose = np.array(pose)
    return pose

def smplh_pose_to_mesh(pose: np.ndarray, mano_model: smplx.SMPLH):
    body_pose = pose[1:22]
    left_hand_pose = pose[22:37]
    right_hand_pose = pose[37:156]
    
    body_pose = torch.tensor(body_pose.flatten(), dtype=torch.float32).unsqueeze(0)
    left_hand_pose = torch.tensor(left_hand_pose.flatten(), dtype=torch.float32).unsqueeze(0)
    right_hand_pose = torch.tensor(right_hand_pose.flatten(), dtype=torch.float32).unsqueeze(0)
    
    output = mano_model(
        body_pose=body_pose, 
        left_hand_pose=left_hand_pose, 
        right_hand_pose=right_hand_pose
    )
    
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = mano_model.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

class SMPLHRenderer:
    body_nodes: Dict[int, pyrender.Node]
    def __init__(self, scene: pyrender.Scene, model: mujoco.MjModel, data: mujoco.MjData):
        self.scene = scene
        self.model = model
        self.data = data
        model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/models/SMPLH_male.pkl'
        self.mano_model = smplx.SMPLH(model_path, ext='npz', use_pca=False, create_transl=False)
        self.body_nodes = {0: pyrender.Node(mesh=pyrender.Mesh.from_trimesh(smplh_pose_to_mesh(np.zeros((154,)), self.mano_model)), matrix=np.eye(4))}
        self.scene.add_node(self.body_nodes[0])
    def __call__(self, matrix_pose: np.ndarray, qpos: np.ndarray):
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)
        qpos = smplh_pose_from_mujoco(self.model, self.data)
        mesh = smplh_pose_to_mesh(qpos, self.mano_model)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        self.scene.remove_node(self.body_nodes[0])
        self.body_nodes[0] = pyrender.Node(mesh=pyrender_mesh, matrix=matrix_pose)
        self.scene.add_node(self.body_nodes[0])
        
def smplh_renderer(scene: pyrender.Scene):
    file_path = os.path.join(os.path.dirname(__file__), "smplh", "smplh.xml")
    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SMPLHRenderer(scene, model, data)


if __name__ == "__main__":
    scene = pyrender.Scene()
    renderer = smplh_renderer(scene)
    renderer(np.eye(4))
    pyrender.Viewer(scene, use_raymond_lighting=True)
