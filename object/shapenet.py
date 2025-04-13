import os
import random
from tqdm import tqdm
import trimesh

SHAPENET_DIRECTORY = "/Users/vineethyeevani/Documents/ShapeNetCore"
class ShapeNet:
    def __init__(self, directory: str):
        self.directory = directory
        self.obj_files = self._get_all_obj_files()

    def _get_all_obj_files(self):
        obj_files = []
        for root, _, files in tqdm(os.walk(self.directory), desc="Walking through shapenet"):
            obj_files.extend([os.path.join(root, file) for file in files if file == 'model_normalized.obj'])
        return obj_files

    def get_random_mesh(self) -> trimesh.Trimesh:
        return trimesh.load(random.choice(self.obj_files), force='mesh')
