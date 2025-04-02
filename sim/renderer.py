from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pyrender
import trimesh

from agent.robot import ArmRenderer
from sim.environment import Environment

@dataclass
class Renderer:
    renderer: pyrender.OffscreenRenderer
    scene: pyrender.Scene
    camera_intrinsics: List[np.ndarray]
    camera_nodes: List[pyrender.Node]
    object_nodes: Dict[int, pyrender.Node]
    arm_nodes: List[pyrender.Node]
    arm_renderers: Dict[int, ArmRenderer]
    gripper_speed: float = 0.1

    def __init__(self, scene: pyrender.Scene, object_meshes: List[trimesh.Trimesh], arm_renderers: Dict[int, ArmRenderer], num_cameras: int, image_width: int = 480, image_height: int = 480):
        num_objects = len(object_meshes)
        yfov = np.pi/4.0
        fx = image_width / (2 * np.tan(yfov / 2))
        fy = image_height / (2 * np.tan(yfov / 2))
        camera_intrinsics = [
            np.array([
                [fx, 0, image_width / 2],
                [0, fy, image_height / 2],
                [0, 0, 1]
            ])
            for _ in range(num_cameras)
        ]

        camera_nodes = [
            pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=image_width / image_height), matrix=np.eye(4))
            for _ in range(num_cameras)
        ]
        object_nodes = {i: pyrender.Node(mesh=pyrender.Mesh.from_trimesh(object_meshes[i]), matrix=np.eye(4)) for i in range(num_objects)}
        [scene.add_node(camera_node) for camera_node in camera_nodes]
        [scene.add_node(object_node) for object_node in object_nodes.values()]

        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

        self.renderer = renderer
        self.scene = scene
        self.camera_intrinsics = camera_intrinsics
        self.camera_nodes = camera_nodes
        self.object_nodes = object_nodes
        self.arm_renderers = arm_renderers

        def get_node_tree(node):
            children = [node]
            for child in node.children:
                children.extend(get_node_tree(child))
            return children
        self.arm_nodes = [child for arm_renderer in self.arm_renderers.values() for node in arm_renderer.body_nodes.values() for child in get_node_tree(node)]

    def __call__(self, env: Environment):
        for obj_id, obj_state in env.object_states.items():
            self.object_nodes[obj_id].matrix = obj_state.pose
        
        # Update arm renderers with joint states
        for arm_id, arm_renderer in self.arm_renderers.items():
            if arm_id in env.robot_states:
                robot_state = env.robot_states[arm_id]
                arm_renderer(robot_state.arm_pose, robot_state.joint_angle)
        
        observations = []
        for cam_idx, camera_state in enumerate(env.camera_states):
            camera_node = self.camera_nodes[cam_idx]
            camera_node.matrix = camera_state.pose
            self.scene.main_camera_node = camera_node
            
            # Standard color and depth rendering
            flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            color, depth = self.renderer.render(self.scene, flags=flags)
            mask = (depth > 0).astype(np.float32)

            seg_node_map = {}
            seg_node_map.update({node: np.array([255, 0, 0]) for node in self.arm_nodes})
            seg_node_map.update({node: np.array([0, 255, 0]) for node in self.object_nodes.values()})
            seg_color, _ = self.renderer.render(
                self.scene,  
                flags=pyrender.RenderFlags.SEG,
                seg_node_map=seg_node_map,
            )
            seg = np.zeros(seg_color.shape[:2], dtype=np.uint8)
            seg[seg_color[:, :, 0] == 255] = 1
            seg[seg_color[:, :, 1] == 255] = 2

            frame_observations = {
                'color': color,
                'depth': depth,
                'mask': mask,
                'seg': seg,
                'camera_intrinsics': self.camera_intrinsics[cam_idx],
                'camera_pose': camera_node.matrix
            }
            observations.append(frame_observations)
        return observations