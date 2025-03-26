import rerun as rr
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def make_image(df, entity, row):
    image_bytes = bytes(df[row][f"{entity}:ImageBuffer"][0][0].to_list())
    image_format = rr.datatypes.ImageFormat(**(df[row][f"{entity}:ImageFormat"][0][0]))
    if image_format.color_model:
        mode = str(image_format.color_model)
    else:
        mode = "F"
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

recording = rr.dataframe.load_recording("dataset.rrd")
schema = recording.schema()
view = recording \
    .view(index="frame", contents="/**")
arrow_table = view.select().read_all()
df = pl.from_arrow(arrow_table)
print(df.columns)


print(make_transform(df, "/world/arm_0/pose", 0))
print(df[0]["/world/arm_0/object_id:Scalar"][0])
print(make_pinhole(df, "/world/3", 0))

# image = make_image(df, "/world/3/color", 0)
# depth = make_image(df, "/world/3/depth", 0)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Color Image")
# plt.imshow(np.array(image), cmap='gray' if image.mode == 'L' else None)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Depth Image")
# plt.imshow(np.array(depth), cmap='gray')
# plt.axis('off')

# plt.show()