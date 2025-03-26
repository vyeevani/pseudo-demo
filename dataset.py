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

recording = rr.dataframe.load_recording("dataset.rrd")
schema = recording.schema()
view = recording \
    .view(index="frame", contents="/**")
arrow_table = view.select().read_all()
df = pl.from_arrow(arrow_table)

image = make_image(df, "/world/3/color", 0)
depth = make_image(df, "/world/3/depth", 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Color Image")
plt.imshow(np.array(image), cmap='gray' if image.mode == 'L' else None)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Depth Image")
plt.imshow(np.array(depth), cmap='gray')
plt.axis('off')

plt.show()