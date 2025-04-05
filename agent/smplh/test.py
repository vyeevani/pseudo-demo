import mujoco
import os

import mujoco.viewer

# Path to the humanoid.xml file
xml_path = os.path.join(os.path.dirname(__file__), 'smplh.xml')

# Load the model from the xml file
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create a viewer to visualize the simulation
viewer = mujoco.viewer.launch(model, data)