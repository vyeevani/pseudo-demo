import numpy as np
import mujoco
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from robot_descriptions import widow_mj_description
from dataclasses import dataclass
from typing import Dict

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

def add_widowx_to_scene(scene: pyrender.Scene):
    model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_nodes = body_nodes_from_model(model, widow_mj_description.PACKAGE_PATH + "/assets")
    for node in body_nodes.values():
        scene.add_node(node)
    return Arm(model, data, body_nodes)

@dataclass
class Arm:
    model: mujoco.MjModel
    data: mujoco.MjData
    body_nodes: Dict[int, pyrender.Node]
    def update_body_nodes(self):
        for body_id, body_node in self.body_nodes.items():
            body_transform = np.eye(4)
            body_transform[:3, :3] = Rotation.from_quat(self.data.xquat[body_id], scalar_first=True).as_matrix()
            body_transform[:3, 3] = self.data.xpos[body_id]
            body_node.matrix = body_transform

scene = pyrender.Scene()
arm = add_widowx_to_scene(scene)
arm.update_body_nodes()

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)