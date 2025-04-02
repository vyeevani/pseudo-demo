import smplx
import trimesh
import torch
import numpy as np
import pyrender
import mujoco
import os
from agent.robot import ArmController, MujocoRenderer
from agent.smplh import smplh_controller

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

def smplh_pose_from_mujoco(model: mujoco.MjModel, data: mujoco.MjData):
    pose = np.zeros((len(SMPLH_BONE_ORDER_NAMES), 3))
    for i, name in enumerate(SMPLH_BONE_ORDER_NAMES):
        pose[i] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
    return pose

def smplh_pose_to_mesh(pose: np.ndarray):
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

def render_smplh_mesh(mesh: trimesh.Trimesh):
    # Create a scene
    scene = pyrender.Scene()

    # Create a mesh object for pyrender
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Add the mesh to the scene
    scene.add(pyrender_mesh)

    # Set up the viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

# Example usage
model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/models/SMPLH_female.pkl'
mano_model = smplx.SMPLH(model_path, ext='npz', use_pca=False, create_transl=False)

file_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/smplh.xml'
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

print(data.qpos[:model.nq])
controller = smplh_controller()
qpos, arm_pose = controller(controller.pose, 0.0)
data.qpos[:model.nq] = qpos
print(data.qpos[:model.nq])
mujoco.mj_forward(model, data)

pose = smplh_pose_from_mujoco(model, data)
# print(pose)
mesh = smplh_pose_to_mesh(pose)
render_smplh_mesh(mesh)

