import os
from tqdm import tqdm
import rerun as rr
from google.cloud import storage
from tqdm import tqdm

import polars as pl
import jax.numpy as jnp
from typing import List, Callable

import jax
import jaxtyping
import equinox
import einops

import grain.python as grain

class AsyncLazyFrame(equinox.Module):
    cols: List[pl.Expr] = equinox.field(static=True)
    fn: Callable = equinox.field(static=True)
    
    def __init__(self, cols: List[pl.Expr], fn: Callable):
        self.cols = cols
        self.fn = fn
            
def partition(arr, lens):
    indices = [sum(lens[:i]) for i in range(len(lens) + 1)]
    return [arr[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]

def resolve_lazy_frames(df: pl.LazyFrame, async_frames: jaxtyping.PyTree[AsyncLazyFrame], rows: pl.Expr):
    async_frames, tree_def = jax.tree.flatten(async_frames, is_leaf=lambda x: isinstance(x, AsyncLazyFrame))
    cols = [col for lazy_frame in async_frames for col in lazy_frame.cols]
    df = df.select(cols + ["episode_frame_number"]).filter(rows)
    df = df.collect()
    frames = [async_frame.fn(df.select(async_frame.cols)) for async_frame in async_frames]
    results = jax.tree.unflatten(tree_def, frames)
    return results
    
def make_images(entity, device=None):
    def fn(df: pl.DataFrame):
        num_rows = df.shape[0]
        image_buffers = []
        for i in range(num_rows):
            image_format = rr.datatypes.ImageFormat(**(df[f"{entity}:ImageFormat"][i][0]))
            image_buffer = df[f"{entity}:ImageBuffer"].to_numpy()[i][0]
            image_buffer = jax.numpy.frombuffer(bytes(image_buffer), dtype=image_format.channel_datatype.to_np_dtype())
            image_buffer = einops.rearrange(image_buffer, "(h w c) -> h w c", h=image_format.height, w=image_format.width)
            if device is not None:
                image_buffer = jax.device_put(image_buffer, device)
            
            image_buffers.append(image_buffer)
        
        image_buffers, _ = einops.pack(image_buffers, "* h w c")
        if image_buffers.shape[-1] == 1:
            image_buffers = einops.rearrange(image_buffers, "b h w 1 -> b h w")
            
        return image_buffers
    return AsyncLazyFrame([pl.col(f"{entity}:ImageBuffer"), pl.col(f"{entity}:ImageFormat")], fn)

def make_transform(entity, device=None):
    def fn(df: pl.DataFrame):
        rotation = df[f"{entity}:TransformMat3x3"].to_numpy()
        rotation, _ = einops.pack(rotation, "* d")
        rotation = jnp.array(rotation)
        rotation = einops.rearrange(rotation, "b (h w) -> b h w", h=3, w=3)
        
        translation = df[f"{entity}:Translation3D"].to_numpy()
        translation, _ = einops.pack(translation, "* d")
        translation = jnp.array(translation)
        translation = einops.rearrange(translation, "b d -> b 1 d")
        
        transform, _ = einops.pack([rotation, translation], "b * a")
        last_row = einops.repeat(jnp.array([0, 0, 0, 1]), "d -> b d", b=transform.shape[0])
        transform, _ = einops.pack([transform, last_row], "b a *")
        if device is not None:
            transform = jax.device_put(transform, device)
        return transform
    return AsyncLazyFrame([pl.col(f"{entity}:TransformMat3x3"), pl.col(f"{entity}:Translation3D")], fn)

def make_scalars(entity, device=None):    
    def fn(df: pl.DataFrame):
        return df[f"{entity}:Scalar"].to_numpy()
    return AsyncLazyFrame([pl.col(f"{entity}:Scalar")], fn)

def get_meta_episode(df: pl.LazyFrame, meta_episode_number: int):
    return df.filter(pl.col("meta_episode_number") == meta_episode_number)

def get_episode(df: pl.LazyFrame, episode_number: int):
    return df.filter(pl.col("episode_number") == episode_number)

def make_timestep(df: pl.LazyFrame, frame_id: int):
    return df.filter(pl.col("frame_id") == frame_id)

def make_rr_dataset(path):
    df = pl.scan_parquet(path)
    return df

class PseudoDemoDatasource(grain.RandomAccessDataSource):
    path: str

    def __init__(self, path):
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Downloading from gs://pseudo_demo-dev...")
            os.makedirs(path, exist_ok=True)
            try:
                # Initialize a storage client
                storage_client = storage.Client()
                
                # Get the bucket
                bucket = storage_client.bucket("pseudo_demo-dev")
                
                # List all blobs in the bucket and download them
                blobs = bucket.list_blobs()
                blobs = list(blobs)
                for blob in tqdm(blobs, desc="Downloading files", unit="file", total=len(blobs)):
                    # Create subdirectories if needed
                    destination_path = os.path.join(path, blob.name)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    
                    # Download the blob
                    blob.download_to_filename(destination_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download data from gs://pseudo_demo-dev: {e}")
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, idx, device=None):
        # start_time = time.time()
        if device is None:
            device = jax.device_get(jax.devices("cpu")[0])
        df = pl.scan_parquet(f"{self.path}/meta_episode_number={idx}/*.parquet")
        meta_episode = get_meta_episode(df, idx)
        # end_time = time.time()
        # print(f"Retrieved meta_episode {idx} in {end_time - start_time:.4f} seconds")
        return meta_episode

def make_timestep_fn(device, downsample, episode_length):
    def make_timestep(episode: pl.LazyFrame):
        # start_time = time.time()
        images = []
        object_masks = []
        gripper_pose = []
        gripper_closed = []
        
        # start_time = time.time()

        # Collect all lazy frames and functions
        async_obs = {
            "images_color": make_images("/world/camera_0/color", device),
            "images_mask": make_images("/world/camera_0/mask", device),
        }
        obs_frames = resolve_lazy_frames(episode, async_obs, pl.col("episode_frame_number").is_in(list(range(0, episode_length, downsample))))
        
        async_act = {
            "gripper_pose": make_transform("/world/arm_0/eef_pose", device),
            "gripper_closed": make_scalars("/world/arm_0/object_id", device)
        }
        act_frames = resolve_lazy_frames(episode, async_act, pl.col("episode_frame_number").is_in(list(range(0, episode_length))))

        # Call the functions with resolved values
        images, _ = einops.pack(obs_frames["images_color"], "* h w c")
        object_masks, _ = einops.pack(obs_frames["images_mask"], "* h w")
        gripper_pose = einops.rearrange(act_frames["gripper_pose"], "(t d) ... -> t d ...", d=downsample)
        gripper_closed = jax.numpy.array([0 if gc is None else 1 for gc in act_frames["gripper_closed"]])
        gripper_closed = einops.rearrange(gripper_closed, "(t d) ... -> t d ...", d=downsample)
        
        output = {
            "obs.images": images,
            "obs.object_masks": object_masks,
            "act.eef": gripper_pose,
            "act.closed": gripper_closed,
        }
        
        # end_time = time.time()
        # print(f"finished processing episode in {end_time - start_time} seconds")
        
        return output
    return make_timestep

def make_dataset(datasource: PseudoDemoDatasource, episode_length=75, downsample=1):
    meta_episode_dataset = grain.MapDataset.source(datasource)
    demo_episode_dataset = meta_episode_dataset.map(lambda df: get_episode(df, 0))
    demo_episode_dataset = demo_episode_dataset.map(make_timestep_fn(jax.device_get(jax.devices("cpu")[0]), downsample, episode_length))
    execution_episode_dataset = meta_episode_dataset.map(lambda df: get_episode(df, 1))
    execution_episode_dataset = execution_episode_dataset.map(make_timestep_fn(jax.device_get(jax.devices("cpu")[0]), downsample, episode_length))
    dataset = grain.experimental.ZipMapDataset([demo_episode_dataset, execution_episode_dataset])
    dataset = dataset.map(lambda demo_and_execution: jax_utils.tree_pack(demo_and_execution, {"obs.images": "* h w c", "obs.object_masks": "* h w", "act.eef": "* d a b", "act.closed": "* d a"})[0])
    return dataset