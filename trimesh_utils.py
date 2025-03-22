import trimesh
import numpy as np

def object_point_transform(object_mesh: trimesh.Trimesh, forward_vector: np.ndarray) -> np.ndarray:
    points, face_indicies = object_mesh.sample(1, return_index=True)
    point, face_index = points[0], face_indicies[0]
    face_normal = object_mesh.face_normals[face_index]
    face_transform = trimesh.geometry.align_vectors(forward_vector, face_normal)
    face_transform[:3, 3] = point
    return face_transform