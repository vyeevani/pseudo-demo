import re
import polars as pl
from polars.datatypes import Struct, List as PlList
from typing import Mapping, List

# Polars dtype → Protobuf type map
_PRIM_MAP = {
    pl.Int8:    "int32",
    pl.Int16:   "int32",
    pl.Int32:   "int32",
    pl.Int64:   "int64",
    pl.UInt8:   "uint32",
    pl.UInt16:  "uint32",
    pl.UInt32:  "uint32",
    pl.UInt64:  "uint64",
    pl.Float32: "float",
    pl.Float64: "double",
    pl.Boolean: "bool",
    pl.Utf8:    "string",
    pl.Binary:  "bytes",
}

def sanitize_identifier(name: str) -> str:
    # replace slashes with underscore, then non-alphanumeric/_ → underscore
    s = name.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    # strip leading chars until letter
    s = re.sub(r"^[^A-Za-z]+", "", s)
    return s or "field"


def _collect_proto_defs(
    schema: Mapping[str, pl.DataType],
    msg_name: str
) -> List[str]:
    """
    Collect .proto definitions for msg_name and nested types, parent-first.
    """
    msg_clean = sanitize_identifier(msg_name)
    lines = [f"message {msg_clean} {{"]
    defs: List[str] = []

    for i, (col, dtype) in enumerate(schema.items(), start=1):
        col_clean = sanitize_identifier(col)
        if isinstance(dtype, PlList):
            inner = dtype.inner
            if isinstance(inner, Struct):
                child = col_clean
                lines.append(f"  repeated {child} {col_clean} = {i};")
                child_schema = {f.name: f.dtype for f in inner.fields}
                defs.extend(_collect_proto_defs(child_schema, child))
            else:
                proto_t = _PRIM_MAP.get(type(inner), "string")
                lines.append(f"  repeated {proto_t} {col_clean} = {i};")

        elif isinstance(dtype, Struct):
            child = col_clean
            lines.append(f"  {child} {col_clean} = {i};")
            child_schema = {f.name: f.dtype for f in dtype.fields}
            defs.extend(_collect_proto_defs(child_schema, child))

        else:
            proto_t = _PRIM_MAP.get(type(dtype), "string")
            lines.append(f"  {proto_t} {col_clean} = {i};")

    lines.append("}")
    return ["\n".join(lines)] + defs


def polars_schema_to_proto(
    df: pl.DataFrame,
    message_name: str = "MyMessage"
) -> str:
    """
    Convert a Polars DataFrame schema into Protobuf .proto definitions,
    including nested messages for Struct and List types.
    """
    all_defs = _collect_proto_defs(df.schema, message_name)
    return "\n\n".join(all_defs)


def normalize(s: str) -> str:
    """Trim whitespace for normalized comparison."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


if __name__ == "__main__":
    # Embedded tests for generator correctness
    tests = [
        (
            "Primitives",
            pl.DataFrame({
                "i32": pl.Series([1],     dtype=pl.Int32),
                "f64": pl.Series([1.0],   dtype=pl.Float64),
                "flag": pl.Series([True], dtype=pl.Boolean),
                "text": pl.Series(["x"],dtype=pl.Utf8),
                "blob": pl.Series([b"x"],dtype=pl.Binary),
            }),
            """
            message Primitives {
              int32 i32 = 1;
              double f64 = 2;
              bool flag = 3;
              string text = 4;
              bytes blob = 5;
            }
            """
        ),
        (
            "ListPrim",
            pl.DataFrame({
                "values": pl.Series([[1,2,3]], dtype=pl.List(pl.Int64)),
            }),
            """
            message ListPrim {
              repeated int64 values = 1;
            }
            """
        ),
        (
            "SimpleStruct",
            pl.DataFrame({
                "meta": pl.Series(
                    [{"a":7, "b":"x"}],
                    dtype=pl.Struct({"a": pl.Int32, "b": pl.Utf8})
                )
            }),
            """
            message SimpleStruct {
              meta meta = 1;
            }

            message meta {
              int32 a = 1;
              string b = 2;
            }
            """
        ),
        (
            "PointList",
            pl.DataFrame({
                "points": pl.Series(
                    [[{"x":1.0,"y":2.0},{"x":3.0,"y":4.0}]],
                    dtype=pl.List(pl.Struct({"x": pl.Float32, "y": pl.Float32}))
                )
            }),
            """
            message PointList {
              repeated points points = 1;
            }

            message points {
              float x = 1;
              float y = 2;
            }
            """
        ),
        (
            "Complex",
            pl.DataFrame({
                "data": pl.Series(
                    [{
                        "name": "foo",
                        "inner": {"flag":True, "nums":[5,10]},
                        "tags": ["a","b"]
                    }],
                    dtype=pl.Struct({
                        "name": pl.Utf8,
                        "inner": pl.Struct({
                            "flag": pl.Boolean,
                            "nums":  pl.List(pl.UInt16)
                        }),
                        "tags": pl.List(pl.Utf8)
                    })
                )
            }),
            """
            message Complex {
              data data = 1;
            }

            message data {
              string name = 1;
              inner inner = 2;
              repeated string tags = 3;
            }

            message inner {
              bool flag = 1;
              repeated uint32 nums = 2;
            }
            """
        ),
        (
            "Slash-Test",
            pl.DataFrame({
                "a/b-c": pl.Series([1,2], dtype=pl.Int32),
            }),
            """
            message Slash_Test {
              int32 a_b_c = 1;
            }
            """
        ),
    ]

    # Run all tests
    for name, df, expected in tests:
        result = polars_schema_to_proto(df, name)
        assert normalize(result) == normalize(expected), (
            f"Test {name} failed:\n{result}\nvs\n{expected}"
        )
    print("All tests passed ✅")


import tensorflow as tf
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
    print(episode_df.schema.to_python())
    episode_table = episode_df.to_arrow()
    proto_schema = polars_schema_to_proto(episode_df, message_name="table")
    with open(f"dataset_array_record/table.proto", "w") as f:
        f.write(proto_schema)
    