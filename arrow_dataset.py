import glob
from tqdm import tqdm
import rerun as rr
import jax
import einops
import polars as pl
import io
from PIL import Image
import numpy as np
import os
import pyarrow as pa
import pyarrow.ipc as ipc

def make_image(df: pl.DataFrame, entity: str, row: int):
    image_format = rr.datatypes.ImageFormat(**(df[f"{entity}:ImageFormat"][row][0]))
    image_buffer = df[f"{entity}:ImageBuffer"].to_numpy()[row][0]
    if image_format.channel_datatype == rr.datatypes.ChannelDatatype.F32:
        mode = "F"
    elif image_format.color_model:
        mode = str(image_format.color_model)
    elif image_format.channel_datatype == rr.datatypes.ChannelDatatype.U8: 
        mode = "L"
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    pil_image = Image.frombuffer(mode, (image_format.width, image_format.height), image_buffer)
    return pil_image

def image_to_str(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="tiff")
    return buffer.getvalue()

def compress_image(df: pl.DataFrame, entity: str):
    image_strs = []
    for i in range(df.shape[0]):
        image = make_image(df, entity, i)
        image_strs.append(image_to_str(image))
    df = df.drop([f"{entity}:ImageBuffer", f"{entity}:ImageFormat"])
    df = df.with_columns(pl.Series(name=f"{entity}:ImagePNG", values=image_strs))
    return df

def write_largebinary_ipc(df: pl.DataFrame, output_path: str):
    table = df.to_arrow()
    new_arrays = []

    for col_name in table.schema.names:
        col = table[col_name]
        if col_name.endswith(":ImagePNG"):
            new_col = pa.array(col.to_pylist(), type=pa.large_binary())
        else:
            new_col = col
        new_arrays.append(new_col)

    full_table = pa.table(new_arrays, names=table.schema.names).combine_chunks()

    with pa.OSFile(output_path, 'wb') as f:
        writer = ipc.new_file(f, full_table.schema)

        for i in range(full_table.num_rows):
            row_arrays = [full_table.column(j).chunk(0).slice(i, 1) for j in range(full_table.num_columns)]
            row_batch = pa.RecordBatch.from_arrays(row_arrays, schema=full_table.schema)
            writer.write_batch(row_batch)

        writer.close()


# Create output directory if it doesn't exist
if not os.path.exists("dataset_arrow"):
    os.makedirs("dataset_arrow")
    print("Created directory: dataset_arrow")
else:
    print("Directory already exists: dataset_arrow")

num_recordings = len(glob.glob("dataset_rrd/*.rrd"))
num_recordings = 10
for recording_file in tqdm(list(range(num_recordings)), desc="Processing recording files"):
    recording = rr.dataframe.load_recording(f"dataset_rrd/dataset_{recording_file}.rrd")
    view = recording.view(index="frame_id", contents="/**")
    arrow_table = view.select().read_all()
    episode_df = pl.from_arrow(arrow_table)
    episode_df.sort(["meta_episode_number", "episode_number", "episode_frame_number"])
    episode_df = compress_image(episode_df, "/world/camera_0/color")
    episode_df = compress_image(episode_df, "/world/camera_0/mask")
    print(episode_df.head())
    # episode_df.write_ipc(f"dataset_arrow/dataset_{recording_file}.arrow", compression="uncompressed")
    write_largebinary_ipc(episode_df, f"dataset_arrow/dataset_{recording_file}.arrow")