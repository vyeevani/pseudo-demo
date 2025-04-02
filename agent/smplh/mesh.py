import smplx
import trimesh
import numpy as np
import torch

# Load the MANO model
model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/models/SMPLH_male.pkl'
# model_path = '/Users/vineethyeevani/Documents/pseudo-demo/agent/smplh/model.npz'
mano_model = smplx.SMPLH(model_path, ext='npz')

# # Generate a random pose and shape
# pose = torch.randn(1, 45)
# shape = torch.randn(1, 10)

# # Call the MANO model with the generated pose and shape
# output = mano_model(pose, betas=shape)

# # Extract the vertices and faces of the mesh
# vertices = output.vertices.detach().cpu().numpy().squeeze()
# faces = mano_model.faces

# # Create a trimesh object
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
