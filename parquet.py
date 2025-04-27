import glob
from tqdm import tqdm
import rerun as rr
import jax
import einops
import polars as pl
import io
from PIL import Image
import numpy as np

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
    buffer = list(buffer.getvalue())
    print(len(buffer))
    return buffer
    # return buffer.getvalue()

def compress_image(df: pl.DataFrame, entity: str):
    image_strs = []
    for i in range(df.shape[0]):
        image = make_image(df, entity, i)
        image_strs.append(image_to_str(image))
    df = df.drop([f"{entity}:ImageBuffer", f"{entity}:ImageFormat"])
    df = df.with_columns(pl.Series(name=f"{entity}:ImagePNG", values=image_strs))
    return df

num_recordings = len(glob.glob("dataset_rrd/*.rrd"))
# num_recordings = 10
for recording_file in tqdm(list(range(num_recordings)), desc="Processing recording files"):
    recording = rr.dataframe.load_recording(f"dataset_rrd/dataset_{recording_file}.rrd")
    view = recording.view(index="frame_id", contents="/**")
    arrow_table = view.select().read_all()
    episode_df = pl.from_arrow(arrow_table)
    episode_df.sort(["meta_episode_number", "episode_number", "episode_frame_number"])
    image_strs = []
    
    # episode_df = compress_image(episode_df, "/world/camera_0/color")
    # episode_df = compress_image(episode_df, "/world/camera_0/mask")
    
    episode_df.write_parquet(f"dataset_parquet", row_group_size=1, compression="zstd", partition_by="meta_episode_number")
    # episode_df.write_parquet(f"dataset_parquet", row_group_size=1, compression="uncompressed", partition_by="meta_episode_number")

# print("Loading parquet files from dataset directory...")
# lazy_df = pl.scan_parquet("dataset/**/*.parquet")
# print("Collecting data into memory...")
# df = lazy_df.collect()
# print("Writing combined dataset to dataset.parquet...")
# df.write_parquet("dataset.parquet", compression="zstd")
# print("Successfully wrote dataset.parquet")