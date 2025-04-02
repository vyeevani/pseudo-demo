import smplx
import trimesh
import numpy as np
import torch

# Load the MANO model
model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/models/SMPLH_male.pkl'
# model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/model.npz'
mano_model = smplx.SMPLH(model_path, ext='npz')

# Generate a random pose and shape
pose = torch.randn(1, 10)
body_pose = torch.randn(1, 63)

# Call the MANO model with the generated pose and shape
output = mano_model(pose, body_pose=body_pose)

# Extract the vertices and faces of the mesh
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = mano_model.faces

# Create a trimesh object
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

import pyrender

# Create a scene
scene = pyrender.Scene()

# Create a mesh object for pyrender
mesh = pyrender.Mesh.from_trimesh(mesh)

# Add the mesh to the scene
scene.add(mesh)

# Set up the viewer
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
