import trimesh
import numpy as np

def object_point_and_normal(object_mesh: trimesh.Trimesh) -> np.ndarray:
    points, face_indicies = object_mesh.sample(1, return_index=True)
    point, face_index = points[0], face_indicies[0]
    face_normal = object_mesh.face_normals[face_index]
    return point, face_normal