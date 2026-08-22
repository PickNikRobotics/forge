"""Round-trip tests: LeRobotV3Writer output read back with LeRobotV3Reader."""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import Episode, Frame, LazyImage
from forge.formats.lerobot_v3.reader import LeRobotV3Reader
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _make_episode(episode_idx: int, num_frames: int, task: str) -> Episode:
    def make_image_loader(index: int):
        def loader():
            img = np.full((32, 32, 3), (index * 10) % 256, dtype=np.uint8)
            return img

        return loader

    frames = [
        Frame(
            index=i,
            timestamp=i / 30.0,
            images={
                "camera0": LazyImage(
                    loader=make_image_loader(i), height=32, width=32, channels=3
                )
            },
            state=np.full(3, episode_idx, dtype=np.float32),
            action=np.full(3, episode_idx + 0.5, dtype=np.float32),
        )
        for i in range(num_frames)
    ]
    return Episode(
        episode_id=f"ep_{episode_idx:03d}",
        metadata={"num_frames": num_frames},
        language_instruction=task,
        robot_type="test_robot",
        cameras=["camera0"],
        state_dim=3,
        action_dim=3,
        fps=30.0,
        _frames_cache=frames,
    )


class TestLeRobotV3RoundTrip:
    """Writer output must read back losslessly through the reader."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_language_instruction_survives_round_trip(self, tmp_path: Path):
        """read_episodes restores tasks from v3.0 parquet metadata.

        v3.0 datasets store tasks in meta/tasks.parquet and per-episode
        metadata in meta/episodes/*.parquet (not the v2 jsonl files), so a
        reader that only checks the jsonl paths yields every episode with
        language_instruction=None and round-trips lose the language labels.
        """
        output_dir = tmp_path / "dataset"
        episodes = [
            _make_episode(0, 3, "pick up the red cube"),
            _make_episode(1, 4, "place the cube in the bin"),
        ]
        writer = LeRobotV3Writer(
            LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        )
        writer.write_dataset(iter(episodes), output_dir)

        read_back = list(LeRobotV3Reader().read_episodes(output_dir))

        assert len(read_back) == 2
        assert read_back[0].language_instruction == "pick up the red cube"
        assert read_back[1].language_instruction == "place the cube in the bin"
        assert len(list(read_back[0].frames())) == 3
        assert len(list(read_back[1].frames())) == 4

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_video_frames_survive_round_trip(self, tmp_path: Path):
        """read_episodes decodes each episode's pixels from its own chunk.

        Video chunk files restart at frame 0, but frames carry the global
        dataset index — seeking by the absolute index over-seeks every
        episode outside the first chunk and silently returns black frames.
        """
        import numpy as np

        output_dir = tmp_path / "dataset"
        episodes = [
            _make_episode(0, 5, "first"),
            _make_episode(1, 5, "second"),
        ]
        writer = LeRobotV3Writer(
            LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        )
        writer.write_dataset(iter(episodes), output_dir)

        read_back = list(LeRobotV3Reader().read_episodes(output_dir))

        assert len(read_back) == 2
        for episode_index, episode in enumerate(read_back):
            # Frame 3, not 0: _make_episode's gradient paints frame N with
            # value N * 10, so frame 0 is legitimately black.
            frame = list(episode.frames())[3]
            image = frame.images["camera0"]
            mean = float(np.asarray(image.load()).mean())
            assert mean > 1.0, (
                f"episode {episode_index} decoded black frames (mean={mean})"
            )

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_image_mode_frames_survive_round_trip(self, tmp_path: Path):
        """Image mode (video=False) round-trips through the reader too.

        The reader used to build every frame's `images` dict exclusively
        from `videos/*.mp4`, so an image-mode dataset (inline PNG bytes,
        dtype: "image") read back with zero images per frame, even though
        the reader already detected `storage="image"` at inspect time.
        """
        output_dir = tmp_path / "dataset"
        episodes = [
            _make_episode(0, 5, "first"),
            _make_episode(1, 5, "second"),
        ]
        writer = LeRobotV3Writer(
            LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot", video=False)
        )
        writer.write_dataset(iter(episodes), output_dir)

        read_back = list(LeRobotV3Reader().read_episodes(output_dir))

        assert len(read_back) == 2
        for episode_index, episode in enumerate(read_back):
            frame = list(episode.frames())[3]
            assert "camera0" in frame.images
            image = np.asarray(frame.images["camera0"].load())
            assert image.shape == (32, 32, 3)
            # _make_episode paints frame N with value (N * 10) % 256; frame 3 -> 30.
            assert float(image.mean()) == pytest.approx(30.0, abs=1.0)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_image_mode_tolerates_null_and_path_only_cells(self, tmp_path: Path):
        """Null and path-only HF Image structs must not crash the reader.

        HuggingFace's `Image()` feature stores either embedded bytes or
        `{"bytes": None, "path": ...}` for on-disk images, and a missing
        image surfaces as a null struct. The reader used to assume every
        cell was a dict carrying bytes.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        output_dir = tmp_path / "dataset"
        writer = LeRobotV3Writer(
            LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot", video=False)
        )
        writer.write_dataset(iter([_make_episode(0, 5, "first")]), output_dir)

        external = output_dir / "images" / "frame1.png"
        external.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), 77, dtype=np.uint8)).save(external)

        parquet_path = next(output_dir.glob("data/**/*.parquet"))
        table = pq.read_table(parquet_path)
        key = "observation.images.camera0"
        col_idx = table.schema.get_field_index(key)
        cells = table.column(key).to_pylist()
        cells[0] = None
        cells[1] = {"bytes": None, "path": "images/frame1.png"}
        old_field = table.schema.field(col_idx)
        # HF's own Image storage type: bytes may be null when path is set.
        new_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
        new_field = pa.field(key, new_type, nullable=True, metadata=old_field.metadata)
        table = table.set_column(col_idx, new_field, pa.array(cells, type=new_type))
        pq.write_table(table, parquet_path)

        frames = list(next(iter(LeRobotV3Reader().read_episodes(output_dir))).frames())

        assert "camera0" not in frames[0].images
        path_mean = float(np.asarray(frames[1].images["camera0"].load()).mean())
        inline_mean = float(np.asarray(frames[3].images["camera0"].load()).mean())
        assert path_mean == pytest.approx(77.0, abs=1.0)
        assert inline_mean == pytest.approx(30.0, abs=1.0)
