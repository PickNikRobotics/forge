"""Field extractors — pull canonical numpy arrays / image bytes from decoded
MCAP messages.

Each extractor takes a decoded message (ROS2 dataclass OR protobuf) plus the
field-config (which sub-field, optional dtype) and returns a value that can
land in a Forge `Frame`. We unify ROS2 and Foxglove/Protobuf message shapes by
attribute access — both expose `.position`, `.data`, `.height`, etc.

Returned types:
- Numeric arrays: numpy.ndarray (default float32 unless dtype overridden)
- Image / compressed: dict {"bytes": ..., "encoding": ..., "height": ..., "width": ...}
- Strings (e.g. task_description): str
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from forge.formats.mcap.topic_config import FieldMapping

# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------


def _coerce_numpy(value: Any, dtype: str | None) -> NDArray[Any]:
    np_dtype = np.dtype(dtype) if dtype else np.float32
    arr = np.asarray(value, dtype=np_dtype)
    return arr


# ROS2 JointState: .position, .velocity, .effort (lists/arrays)
# Foxglove proto JointState (schemas.proto.JointState):
#   .joint_positions, .joint_velocities, .joint_efforts (repeated double)
_JOINT_STATE_FIELDS = {
    "position": ("position", "joint_positions"),
    "velocity": ("velocity", "joint_velocities"),
    "effort": ("effort", "joint_efforts"),
}


def _read_attr(msg: Any, candidates: tuple[str, ...]) -> Any:
    for name in candidates:
        if hasattr(msg, name):
            val = getattr(msg, name)
            # protobuf repeated fields are not lists but iterable
            return val
    raise AttributeError(f"none of {candidates} found on {type(msg).__name__}")


# ROS2 JointState names its joints in .name, Foxglove proto in .joint_names.
_JOINT_NAME_FIELDS = ("name", "joint_names")


def extract_joint_names(msg: Any, schema_name: str) -> list[str]:
    """Return the joint names a JointState (ROS2 or proto) lists, or an empty list for any other schema."""
    if schema_name not in _JOINT_STATE_SCHEMAS:
        return []
    return [str(name) for name in _read_attr(msg, _JOINT_NAME_FIELDS)]


def extract_joint_field(msg: Any, field: str, dtype: str | None) -> NDArray[Any]:
    """Pull position / velocity / effort from JointState (ROS2 or proto)."""
    candidates = _JOINT_STATE_FIELDS.get(field, (field,))
    raw = _read_attr(msg, candidates)
    return _coerce_numpy(list(raw), dtype)


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def extract_image(msg: Any, schema_name: str) -> dict[str, Any]:
    """Return a dict describing the image payload.

    For raw images (sensor_msgs/Image, foxglove.RawImage): includes decoded
    h/w/encoding/data so the reader can reshape to (h, w, c).
    For compressed (CompressedImage / CompressedVideo): bytes + format hint.
    """
    out: dict[str, Any] = {"schema": schema_name}

    # Raw image — height/width/encoding/data fields
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "data"):
        out["height"] = int(msg.height)
        out["width"] = int(msg.width)
        out["encoding"] = getattr(msg, "encoding", "rgb8")
        out["data"] = bytes(msg.data)
        out["format"] = "raw"
        return out

    # Compressed image — has format + data
    if hasattr(msg, "format") and hasattr(msg, "data"):
        out["format"] = str(msg.format)
        out["data"] = bytes(msg.data)
        return out

    # Foxglove CompressedVideo — has format + data + frame_id
    if hasattr(msg, "data"):
        out["format"] = getattr(msg, "format", "video")
        out["data"] = bytes(msg.data)
        return out

    raise ValueError(f"don't know how to extract image from {schema_name}")


# ---------------------------------------------------------------------------
# String / task description extraction
# ---------------------------------------------------------------------------


def extract_string(msg: Any) -> str:
    if hasattr(msg, "data"):  # std_msgs/String
        return str(msg.data)
    if hasattr(msg, "value"):  # foxglove.KeyValuePair
        return str(msg.value)
    if hasattr(msg, "key") and hasattr(msg, "value"):
        return f"{msg.key}={msg.value}"
    return repr(msg)


# ---------------------------------------------------------------------------
# Generic numeric (Float*MultiArray.data, etc.)
# ---------------------------------------------------------------------------


def extract_numeric(msg: Any, field: str | None, dtype: str | None) -> NDArray[Any]:
    field = field or "data"
    raw = _read_attr(msg, (field,))
    return _coerce_numpy(list(raw) if not isinstance(raw, (int, float)) else [raw], dtype)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


_JOINT_STATE_SCHEMAS = {
    "sensor_msgs/msg/JointState",
    "sensor_msgs/JointState",
    "schemas.proto.JointState",
}
_IMAGE_SCHEMAS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/CompressedImage",
    "foxglove.RawImage",
    "foxglove.CompressedImage",
    "foxglove.CompressedVideo",
}
_STRING_SCHEMAS = {
    "std_msgs/msg/String",
    "std_msgs/String",
    "foxglove.KeyValuePair",
}


def extract(msg: Any, schema_name: str, mapping: FieldMapping) -> Any:
    """Top-level dispatch from (schema_name, FieldMapping) -> extracted value.

    Falls back to attribute lookup for unknown schemas.
    """
    if schema_name in _JOINT_STATE_SCHEMAS:
        sub = mapping.field or "position"
        return extract_joint_field(msg, sub, mapping.dtype)
    if schema_name in _IMAGE_SCHEMAS:
        return extract_image(msg, schema_name)
    if schema_name in _STRING_SCHEMAS:
        return extract_string(msg)
    # Generic fallback: treat as numeric array on `mapping.field` (default "data").
    try:
        return extract_numeric(msg, mapping.field, mapping.dtype)
    except (AttributeError, TypeError, ValueError):
        # Last-resort: attribute fetch as-is.
        if mapping.field and hasattr(msg, mapping.field):
            return getattr(msg, mapping.field)
        return msg
