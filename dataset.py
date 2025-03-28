import rerun as rr
import polars as pl
import numpy as np
from PIL import Image

def make_image(df, entity, row):
    image_bytes = bytes(df[row][f"{entity}:ImageBuffer"][0][0])
    image_format = rr.datatypes.ImageFormat(**(df[row][f"{entity}:ImageFormat"][0][0]))
    print(image_format)
    if image_format.channel_datatype == rr.datatypes.ChannelDatatype.F32:
        mode = "F"
    elif image_format.color_model:
        mode = str(image_format.color_model)
    image_buffer = np.frombuffer(image_bytes, image_format.channel_datatype.to_np_dtype())
    image = Image.frombuffer(mode, (image_format.width, image_format.height), image_buffer)
    return image

def make_transform(df, entity, row):
    rotation = np.array(df[row][f"{entity}:TransformMat3x3"][0][0]).reshape(3, 3)
    translation = np.array(df[row][f"{entity}:Translation3D"][0][0])
    transform = np.concatenate([rotation, translation.reshape(-1, 1)], axis=1)
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform

def make_pinhole(df, entity, row):
    intrinsics = np.array(df[row][f"{entity}:PinholeProjection"][0][0]).reshape(3, 3)
    transform = make_transform(df, entity, row)
    return intrinsics, transform

def make_scalars(df, entity, row):
    return df[row][f"{entity}:Scalar"][0]

def make_tensor(df, entity, row):
    return np.array(df[row][f"{entity}:Tensor"][0][0])

def get_meta_episode(df: pl.DataFrame, meta_episode_number: int):
    return df.filter(pl.col("meta_episode_number") == meta_episode_number)

def get_episode(df: pl.DataFrame, episode_number: int):
    return df.filter(pl.col("episode_number") == episode_number)

def make_timestep(df: pl.DataFrame, frame_id: int):
    return df.filter(pl.col("frame_id") == frame_id)

def make_rr_dataset(path):
    recording = rr.dataframe.load_recording(path)
    view = recording.view(index="frame_id", contents="/**")
    arrow_table = view.select().read_all()
    df = pl.from_arrow(arrow_table)
    return df

df = make_rr_dataset("dataset.rrd")
print(df.columns)

import matplotlib.pyplot as plt

image = make_image(df, "/world/0/mask", 0)
print(image)
plt.imshow(np.array(image))
plt.axis('off')
plt.show()
# print(np.array(make_scalars(df, "/world/arm_0/joint_angle", 0)))
# print(get_meta_episode(df, 0))
# print(get_episode(get_meta_episode(df, 0), 0))
# episode = get_episode(get_meta_episode(df, 0), 0)
# print(make_image(episode, "/world/0/color", 0))
# print(make_image(episode, "/world/0/depth", 0))