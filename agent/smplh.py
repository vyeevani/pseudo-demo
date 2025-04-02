from agent.robot import ArmController, MujocoRenderer
import mujoco
import pyrender
import numpy as np
import os

def smplh_controller():
    file_path = os.path.join(os.path.dirname(__file__), "smplh", "smplh.xml")
    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmController(model, data, "L_Wrist", [])

def smplh_renderer(scene: pyrender.Scene):
    file_path = os.path.join(os.path.dirname(__file__), "smplh", "smplh.xml")
    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoRenderer(scene, model, data, "L_Wrist", asset_path=os.path.join(os.path.dirname(__file__), "assets"), mesh_extension="obj")

if __name__ == "__main__":
    scene = pyrender.Scene()
    renderer = smplh_renderer(scene)
    renderer(np.eye(4))
    pyrender.Viewer(scene, use_raymond_lighting=True)
