import numpy as np
import mujoco
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from robot_descriptions import widow_mj_description

def make_visual_nodes(m, asset_path):
    body_to_nodes = {}

    for geom_id in range(m.ngeom):
        # Get the geom using direct ID access
        geom = m.geom(geom_id)
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH and geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        
        parent_body_id = geom.bodyid[0]
        pos = geom.pos
        quat = geom.quat

        # print(f"geom_id: {geom.id}, bodyid: {parent_body_id}, pos: {np.round(pos, 3)}, quat: {np.round(quat, 3)}, sameframe: {geom.sameframe}")

        rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = pos

        if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            # Create a box geometry using trimesh
            size = geom.size
            box = trimesh.creation.box(extents=size * 2)  # size is half extents in MuJoCo
            pyrender_box = pyrender.Mesh.from_trimesh(box)
            node = pyrender.Node(mesh=pyrender_box, matrix=transform)
        else:
            mesh_id = geom.dataid[0]  # Get associated mesh index
            if mesh_id < 0:  # Ensure it's valid
                continue
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            mesh_path = f"{asset_path}/{mesh_name}.stl"
            mesh = trimesh.load(mesh_path)
            scale = m.mesh_scale[mesh_id]
            mesh.apply_scale(scale)
            mesh_pos = m.mesh_pos[mesh_id]
            mesh_quat = m.mesh_quat[mesh_id]
            mesh_rotation = Rotation.from_quat(mesh_quat, scalar_first=True).as_matrix()
            mesh_transform = np.eye(4)
            mesh_transform[:3, :3] = mesh_rotation
            mesh_transform[:3, 3] = mesh_pos
            mesh_transform = np.linalg.inv(mesh_transform)
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            mesh_node = pyrender.Node(mesh=pyrender_mesh, matrix=mesh_transform)
            node = pyrender.Node(matrix=transform, children=[mesh_node])
        if parent_body_id not in body_to_nodes:
            body_to_nodes[parent_body_id] = []
        body_to_nodes[parent_body_id].append(node)
    return body_to_nodes

def create_body_nodes(m):
    body_to_nodes = {}
    for body_id in range(m.nbody):
        node = pyrender.Node()
        body_to_nodes[body_id] = node
    return body_to_nodes

def update_body_nodes(body_nodes, d):
    for body_id, body_node in body_nodes.items():
        body_transform = np.eye(4)
        body_transform[:3, :3] = Rotation.from_quat(d.xquat[body_id], scalar_first=True).as_matrix()
        body_transform[:3, 3] = d.xpos[body_id]
        body_node.matrix = body_transform
    return body_nodes

asset_path = widow_mj_description.PACKAGE_PATH + "/assets"
model = mujoco.MjModel.from_xml_path(widow_mj_description.MJCF_PATH)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

scene = pyrender.Scene()
visual_meshes = make_visual_nodes(model, asset_path)
bodies = create_body_nodes(model)

for body_id, body_node in bodies.items():
    scene.add_node(body_node)

bodies = update_body_nodes(bodies, data)

for parent_body_id, visual_nodes in visual_meshes.items():
    for visual_node in visual_nodes:
        scene.add_node(visual_node, bodies[parent_body_id])

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
