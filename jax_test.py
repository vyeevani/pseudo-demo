import jax
import equinox
import polars as pl
import rerun as rr
import glob
from tqdm import tqdm
import numpy as np
import os

if not os.path.exists("dataset_array_record"):
    os.makedirs("dataset_array_record")
    print("Created directory: dataset_array_record")
else:
    print("Directory already exists: dataset_array_record")
    
num_recordings = len(glob.glob("dataset_rrd/*.rrd"))
for episode_number in tqdm(range(num_recordings), desc="Processing recording files"):
    recording_file = f"dataset_rrd/dataset_{episode_number}.rrd"
    recording = rr.dataframe.load_recording(recording_file)
    view = recording.view(index="frame_id", contents="/**")
    arrow_table = view.select().read_all()
    episode_df = pl.from_arrow(arrow_table)
    episode_df.sort(["meta_episode_number", "episode_number", "episode_frame_number"])
    episode_table = episode_df.to_numpy()
    equinox.tree_pprint(episode_table)
    