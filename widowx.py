from robot import ArmController, ArmRenderer
import mujoco
import pyrender
from robot_descriptions import widow_mj_description
import numpy as np

def widowx_controller():
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmController(model, data, "wx250s/gripper_link", ["right_finger", "left_finger"])

def widowx_renderer(scene: pyrender.Scene):
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ArmRenderer(scene, model, data, "wx250s/gripper_link")

if __name__ == "__main__":
    scene = pyrender.Scene()
    renderer = widowx_renderer(scene)
    renderer(np.eye(4))
    pyrender.Viewer(scene, use_raymond_lighting=True)
