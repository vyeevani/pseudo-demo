import rerun as rr
import polars as pl
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List

def make_image(df, entity, row):
    image_bytes = bytes(df[row][f"{entity}:ImageBuffer"][0][0])
    image_format = rr.datatypes.ImageFormat(**(df[row][f"{entity}:ImageFormat"][0][0]))
    print(image_format)
    if image_format.channel_datatype == rr.datatypes.ChannelDatatype.F32:
        mode = "F"
    elif image_format.color_model:
        mode = str(image_format.color_model)
    elif image_format.channel_datatype == rr.datatypes.ChannelDatatype.U8: 
        mode = "L"
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    
    image_buffer = np.frombuffer(image_bytes, image_format.channel_datatype.to_np_dtype())
    image = Image.frombuffer(mode, (image_format.width, image_format.height), image_buffer)
    return image

@dataclass
class Annotation:
    id: int
    label: str

def make_annotation_context(df, entity, row) -> List[Annotation]:
    annotations = []
    for annotation in df[row][f"{entity}:AnnotationContext"][0][0]:
        annotation = annotation["class_description"]["info"]
        annotations.append(Annotation(id=annotation["id"], label=annotation["label"]))
    return annotations

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
for col in df.columns:
    print(col)

image = make_image(df, "/world/0/seg", 0)
print(make_annotation_context(df, "/world/0/seg", 0))

import matplotlib.pyplot as plt

plt.imshow(np.array(image))
plt.axis('off')
plt.show()

# import matplotlib.pyplot as plt

# image = make_image(df, "/world/0/mask", 0)
# print(image)
# plt.imshow(np.array(image))
# plt.axis('off')
# plt.show()
# print(np.array(make_scalars(df, "/world/arm_0/joint_angle", 0)))
# print(get_meta_episode(df, 0))
# print(get_episode(get_meta_episode(df, 0), 0))
# episode = get_episode(get_meta_episode(df, 0), 0)
# print(make_image(episode, "/world/0/color", 0))
# print(make_image(episode, "/world/0/depth", 0))