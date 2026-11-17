"""MCAP format reader for Forge.

Driven by a TopicConfig — loads each configured topic's full stream once,
splits the primary topic's timestamps into episodes per the chosen strategy,
then per-frame aligns secondary streams via sync.align().

Supported message encodings: ROS2 CDR (mcap-ros2-support) + Foxglove Protobuf
(mcap-protobuf-support).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.exceptions import EpisodeNotFoundError, MissingDependencyError
from forge.core.models import (
    CameraInfo,
    DatasetInfo,
    Episode,
    Frame,
    LazyImage,
)
from forge.formats.mcap import episode_split, sync
from forge.formats.mcap.decoders import DecodedMessage, iter_decoded
from forge.formats.mcap.extractors import extract, extract_joint_names
from forge.formats.mcap.inspect import generate_config, inspect_mcap
from forge.formats.mcap.topic_config import FieldMapping, TopicConfig
from forge.formats.mcap.video_decode import decode_packet_stream, looks_like_video_format
from forge.formats.registry import FormatRegistry

logger = logging.getLogger(__name__)


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap",
            feature="MCAP format support",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


@FormatRegistry.register_reader("mcap")
class MCAPReader:
    """Reader for MCAP files (ROS2 CDR + Foxglove Protobuf)."""

    @property
    def format_name(self) -> str:
        return "mcap"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file() and path.suffix.lower() == ".mcap":
            return True
        if path.is_dir() and not (path / "metadata.yaml").exists():
            if any(path.glob("*.mcap")):
                return True
        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        try:
            inv = inspect_mcap(path if path.is_file() else next(path.glob("*.mcap")))
        except Exception:
            return None
        return f"mcap/{inv.profile}" if inv.profile else "mcap"

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def inspect(self, path: Path) -> DatasetInfo:
        inv = inspect_mcap(path if path.is_file() else next(path.glob("*.mcap")))
        info = DatasetInfo(path=Path(path), format="mcap")
        info.format_version = f"mcap/{inv.profile}" if inv.profile else "mcap"
        info.num_episodes = 1
        info.total_frames = inv.total_messages
        info.has_timestamps = inv.duration_ns is not None
        if inv.duration_ns and inv.total_messages:
            duration_sec = inv.duration_ns / 1e9
            if duration_sec > 0:
                info.inferred_fps = round(inv.total_messages / duration_sec, 2)
        info.metadata["mcap_profile"] = inv.profile
        info.metadata["mcap_channels"] = [
            {"topic": c.topic, "schema": c.schema_name, "messages": c.message_count}
            for c in inv.channels
        ]
        return info

    # ------------------------------------------------------------------
    # Read episodes
    # ------------------------------------------------------------------

    def read_episodes(
        self,
        path: Path,
        config: TopicConfig | None = None,
    ) -> Iterator[Episode]:
        """Yield episodes parsed from an MCAP file.

        If `config` is None, generate_config() is called to produce a sensible
        default. If that returns skipped=True, raises ValueError.
        """
        _check_mcap()
        path = Path(path)

        if config is None:
            res = generate_config(path)
            if res.skipped or res.config is None:
                raise ValueError(
                    f"no usable config could be auto-generated for {path}: "
                    f"{res.reason}"
                )
            config = res.config

        if not config.fields:
            raise ValueError("TopicConfig.fields is empty — nothing to read")

        # ---- Load all configured topic streams into memory ----
        topic_to_field_names: dict[str, list[str]] = {}
        for field_name, mapping in config.fields.items():
            topic_to_field_names.setdefault(mapping.topic, []).append(field_name)

        topics_to_load = set(topic_to_field_names.keys())
        marker_topic = config.episodes.marker_topic
        if marker_topic:
            topics_to_load.add(marker_topic)
        task_topic = config.task.topic
        if task_topic:
            topics_to_load.add(task_topic)

        streams: dict[str, list[DecodedMessage]] = {t: [] for t in topics_to_load}
        for m in iter_decoded(path, topics=topics_to_load):
            streams[m.topic].append(m)

        # ---- Pick primary topic & timestamps ----
        primary_field_name = config.sync.primary
        if primary_field_name is None:
            # Pick the most-populated configured field as primary.
            primary_field_name = max(
                config.fields.keys(),
                key=lambda f: len(streams.get(config.fields[f].topic, [])),
            )
        if primary_field_name not in config.fields:
            raise ValueError(
                f"sync.primary {primary_field_name!r} not in config.fields"
            )
        primary_topic = config.fields[primary_field_name].topic
        primary_stream = streams.get(primary_topic, [])
        if not primary_stream:
            raise ValueError(
                f"primary topic {primary_topic!r} has no messages in {path}"
            )
        primary_ts_ns = [m.timestamp_ns for m in primary_stream]

        # ---- Episode boundaries ----
        marker_ts_ns: list[int] | None = None
        if config.episodes.strategy == "marker" and marker_topic:
            marker_ts_ns = [m.timestamp_ns for m in streams.get(marker_topic, [])]

        segment_signal: np.ndarray | None = None
        if config.episodes.strategy == "segment" and config.episodes.segment_signal:
            sig_topic, _, sig_field = config.episodes.segment_signal.partition(".")
            sig_stream = streams.get(sig_topic, [])
            if sig_stream:
                vals = []
                for m in sig_stream:
                    try:
                        vals.append(
                            float(np.asarray(getattr(m.message, sig_field or "data"))[0])
                        )
                    except Exception:
                        vals.append(0.0)
                segment_signal = np.asarray(vals, dtype=np.float64)

        boundaries = episode_split.compute_boundaries(
            primary_ts_ns,
            config.episodes,
            marker_ts_ns=marker_ts_ns,
            segment_signal=segment_signal,
        )

        if not boundaries:
            return

        # ---- Pre-extract every secondary stream's values + timestamps ----
        secondary_data: dict[str, dict[str, list]] = {}
        for field_name, mapping in config.fields.items():
            stream = streams.get(mapping.topic, [])
            ts = [m.timestamp_ns for m in stream]
            vals = [extract(m.message, m.schema_name, mapping) for m in stream]
            self._predecode_video_stream(field_name, ts, vals)
            secondary_data[field_name] = {"ts": ts, "vals": vals, "mapping": mapping}

        # ---- Inferred Episode-level metadata ----
        cameras = self._infer_cameras(secondary_data)
        state_dim = self._infer_dim(secondary_data, "observation.state")
        action_dim = self._infer_dim(secondary_data, "action")
        state_names = self._infer_joint_names(streams, config, "observation.state", state_dim)
        action_names = self._infer_joint_names(streams, config, "action", action_dim)
        fps = self._infer_fps(primary_ts_ns)

        # ---- Task description (one-shot lookup) ----
        language: str | None = config.task.description
        if language is None and task_topic:
            task_stream = streams.get(task_topic, [])
            if task_stream:
                from forge.formats.mcap.extractors import extract_string

                language = extract_string(task_stream[0].message)

        # ---- Build episodes lazily ----
        for ep_idx, (start, end) in enumerate(boundaries):
            ep_primary_ts = primary_ts_ns[start:end]
            episode_id = str(ep_idx)

            def _make_loader(
                ep_primary_ts_local: list[int] = ep_primary_ts,
                secondary_data_local: dict = secondary_data,
                config_local: TopicConfig = config,
            ) -> Any:
                def loader() -> Iterator[Frame]:
                    yield from self._build_frames(
                        ep_primary_ts_local, secondary_data_local, config_local
                    )

                return loader

            ep = Episode(
                episode_id=episode_id,
                metadata={"num_frames": len(ep_primary_ts)},
                language_instruction=language,
                cameras=cameras,
                state_dim=state_dim,
                action_dim=action_dim,
                state_names=state_names,
                action_names=action_names,
                fps=fps,
                _frame_loader=_make_loader(),
            )
            yield ep

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        for ep in self.read_episodes(path):
            if ep.episode_id == episode_id:
                return ep
        raise EpisodeNotFoundError(episode_id, path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_frames(
        self,
        primary_ts_ns: list[int],
        secondary_data: dict[str, dict[str, list]],
        config: TopicConfig,
    ) -> Iterator[Frame]:
        method = config.sync.method
        max_skew_ns = int(config.sync.max_skew_ms * 1e6)
        drop_skew_ns = max_skew_ns * 10
        skew_violations: dict[str, dict[str, Any]] = {}

        last = len(primary_ts_ns) - 1
        for i, t in enumerate(primary_ts_ns):
            frame_extras: dict[str, Any] = {}
            images: dict[str, LazyImage] = {}
            state: np.ndarray | None = None
            action: np.ndarray | None = None
            drop_frame = False

            for field_name, info in secondary_data.items():
                ts = info["ts"]
                vals = info["vals"]
                mapping: FieldMapping = info["mapping"]
                if not ts:
                    continue
                value, skew = sync.align(t, ts, vals, method=method)
                if skew > drop_skew_ns:
                    drop_frame = True
                    break
                if skew > max_skew_ns:
                    bucket = skew_violations.setdefault(
                        field_name, {"count": 0, "max_ns": 0}
                    )
                    bucket["count"] += 1
                    bucket["max_ns"] = max(bucket["max_ns"], skew)

                self._assign_field(
                    field_name=field_name,
                    value=value,
                    mapping=mapping,
                    images=images,
                    frame_extras=frame_extras,
                    out_state=state,
                    out_action=action,
                )
                if field_name == "observation.state" and isinstance(value, np.ndarray):
                    state = value
                elif field_name == "action" and isinstance(value, np.ndarray):
                    action = value

            if drop_frame:
                continue

            yield Frame(
                index=i,
                timestamp=t / 1e9,
                images=images,
                state=state,
                action=action,
                is_first=(i == 0),
                is_last=(i == last),
                extras=frame_extras,
            )

        # End-of-episode summary: one warning per field that exceeded threshold.
        for field_name, stats in skew_violations.items():
            logger.warning(
                "mcap: field %s exceeded sync threshold (%.1fms) on %d frames; max skew %.1fms",
                field_name,
                max_skew_ns / 1e6,
                stats["count"],
                stats["max_ns"] / 1e6,
            )

    def _assign_field(
        self,
        field_name: str,
        value: Any,
        mapping: FieldMapping,
        images: dict[str, LazyImage],
        frame_extras: dict[str, Any],
        out_state: np.ndarray | None,
        out_action: np.ndarray | None,
    ) -> None:
        """Route an extracted value to the right Frame slot."""
        if field_name == "observation.state":
            return  # caller assigns
        if field_name == "action":
            return
        if field_name.startswith("observation.images."):
            cam_name = field_name.removeprefix("observation.images.")
            if isinstance(value, dict) and "data" in value:
                images[cam_name] = self._image_from_dict(cam_name, value)
            return
        # Everything else lands in extras (ee_pose, state.velocity, etc.)
        frame_extras[field_name] = value

    def _image_from_dict(self, cam_name: str, info: dict[str, Any]) -> LazyImage:
        """Construct a LazyImage from an extracted image dict."""
        # Raw image: known h/w/encoding, decode via numpy reshape.
        if info.get("format") == "raw":
            h = info["height"]
            w = info["width"]
            enc = info.get("encoding", "rgb8").lower()
            data = info["data"]
            channels = 3 if ("rgb" in enc or "bgr" in enc) else 1

            def loader(d: bytes = data, h: int = h, w: int = w, c: int = channels) -> np.ndarray:
                arr = np.frombuffer(d, dtype=np.uint8)
                if c == 3:
                    return arr.reshape(h, w, 3)
                return arr.reshape(h, w)

            return LazyImage(loader=loader, height=h, width=w, channels=channels)

        # Compressed image / video: shape unknown until decoded; defer.
        data = info["data"]

        def loader(d: bytes = data) -> np.ndarray:
            try:
                import cv2
            except ImportError:
                # Fallback: return the raw byte buffer as a 1D uint8 array so
                # downstream code can still handle it explicitly.
                return np.frombuffer(d, dtype=np.uint8)
            arr = np.frombuffer(d, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                return arr
            return decoded[..., ::-1]  # BGR -> RGB

        # Best-effort h/w — leave 0 if we can't tell from the dict.
        return LazyImage(loader=loader, height=0, width=0, channels=3)

    def _infer_cameras(
        self, secondary_data: dict[str, dict[str, list]]
    ) -> dict[str, CameraInfo]:
        cameras: dict[str, CameraInfo] = {}
        for field_name, info in secondary_data.items():
            if not field_name.startswith("observation.images."):
                continue
            cam_name = field_name.removeprefix("observation.images.")
            vals = info["vals"]
            if not vals or not isinstance(vals[0], dict):
                continue
            sample = vals[0]
            if sample.get("format") == "raw":
                cameras[cam_name] = CameraInfo(
                    name=cam_name,
                    height=sample["height"],
                    width=sample["width"],
                    channels=3 if "rgb" in sample.get("encoding", "rgb8").lower() else 1,
                    encoding=sample.get("encoding", "rgb"),
                )
            else:
                h, w = self._probe_compressed_dims(sample)
                cameras[cam_name] = CameraInfo(
                    name=cam_name,
                    height=h,
                    width=w,
                    channels=3,
                    encoding="compressed",
                )
        return cameras

    @staticmethod
    def _predecode_video_stream(
        field_name: str, ts: list[int], vals: list[Any]
    ) -> None:
        """If the stream is h264/h265, replace each value's `data` with a
        pre-decoded RGB frame. Mutates `vals` in place. Called once per stream
        before episode build, so codec context state is preserved.

        If decode produces fewer frames than packets (SPS/PPS-only packets),
        we drop the leading timestamps so frames stay aligned by index.
        """
        if not field_name.startswith("observation.images.") or not vals:
            return
        if not isinstance(vals[0], dict):
            return
        if not looks_like_video_format(vals[0].get("format")):
            return

        packets = [v.get("data", b"") for v in vals]
        codec = "hevc" if "h265" in vals[0].get("format", "").lower() or "hevc" in vals[0].get("format", "").lower() else "h264"
        decoded = decode_packet_stream(packets, codec=codec)
        if not decoded:
            logger.warning(
                "mcap: %s — could not decode %s stream (PyAV missing or codec error); "
                "image bytes will be left raw",
                field_name, codec,
            )
            return

        n_drop = len(packets) - len(decoded)
        if n_drop > 0:
            del ts[:n_drop]
            del vals[:n_drop]
        for i, frame in enumerate(decoded):
            vals[i] = {
                "format": "raw",
                "data": frame.tobytes(),
                "height": frame.shape[0],
                "width": frame.shape[1],
                "encoding": "rgb8",
                "schema": vals[i].get("schema", ""),
            }

    @staticmethod
    def _probe_compressed_dims(sample: dict[str, Any]) -> tuple[int, int]:
        """Best-effort decode of a compressed image / video sample to learn h/w.

        For JPEG/PNG we can use cv2.imdecode. For h264-encoded video, decoding
        a single chunk doesn't yield a frame without an SPS/PPS context, so we
        fall back to a (1, 1) sentinel — downstream code should handle this by
        decoding lazily on access.
        """
        data = sample.get("data", b"")
        if not data:
            return 1, 1
        try:
            import cv2
            import numpy as _np

            arr = _np.frombuffer(data, dtype=_np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                return decoded.shape[0], decoded.shape[1]
        except ImportError:
            pass
        except Exception:
            pass
        return 1, 1

    def _infer_dim(
        self, secondary_data: dict[str, dict[str, list]], field_name: str
    ) -> int | None:
        info = secondary_data.get(field_name)
        if not info or not info["vals"]:
            return None
        sample = info["vals"][0]
        if isinstance(sample, np.ndarray):
            return int(sample.size)
        return None

    @staticmethod
    def _infer_joint_names(
        streams: dict[str, list[DecodedMessage]],
        config: TopicConfig,
        field_name: str,
        dim: int | None,
    ) -> list[str] | None:
        """Return the joint names every message of a JointState field lists, or None when none list any.

        Values are read by index, so a stream whose names change between messages is refused.
        """
        mapping = config.fields.get(field_name)
        if mapping is None:
            return None
        topic = mapping.topic
        distinct = {
            tuple(extract_joint_names(m.message, m.schema_name)) for m in streams.get(topic, [])
        }
        if distinct <= {()}:
            return None
        if len(distinct) > 1:
            first, second = sorted(distinct)[:2]
            raise ValueError(
                f"{topic!r} lists joint names {list(first)} in one message and {list(second)} in "
                f"another; {field_name} values are read by index, so record the topic with a fixed "
                "joint list"
            )
        names = list(distinct.pop())
        if len(names) != dim or len(set(names)) != len(names):
            raise ValueError(
                f"{topic!r} lists {len(names)} joint names {names} for {dim} {field_name} values; "
                "each value needs one unique name"
            )
        return names

    def _infer_fps(self, primary_ts_ns: list[int]) -> float | None:
        if len(primary_ts_ns) < 2:
            return None
        duration_sec = (primary_ts_ns[-1] - primary_ts_ns[0]) / 1e9
        if duration_sec <= 0:
            return None
        return round((len(primary_ts_ns) - 1) / duration_sec, 2)
