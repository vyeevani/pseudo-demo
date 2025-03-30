from agent.robot import ArmController, ArmRenderer
import mujoco
import pyrender
import numpy as np

def humanoid_controller():
    model = mujoco.MjModel.from_xml_path("agent/humanoid/humanoid.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmController(model, data, "lh_palm", [])

def humanoid_renderer(scene: pyrender.Scene):
    model = mujoco.MjModel.from_xml_path("agent/humanoid/humanoid.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmRenderer(scene, model, data, "lh_palm", asset_path="agent/humanoid/assets", mesh_extension="obj")

if __name__ == "__main__":
    scene = pyrender.Scene()
    renderer = humanoid_renderer(scene)
    renderer(np.eye(4))
    pyrender.Viewer(scene, use_raymond_lighting=True)
