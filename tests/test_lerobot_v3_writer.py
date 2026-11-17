"""Tests for LeRobot V3 writer."""

import json
from pathlib import Path

import numpy as np
import pytest

from forge.core.exceptions import ConversionError
from forge.core.models import CameraInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _check_image_mode_dependencies_available() -> bool:
    """Check dependencies for image mode (no `av` — video encoder isn't used)."""
    try:
        import PIL  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample episode for testing."""

    def create_frame(idx: int) -> Frame:
        def make_image_loader(index: int):
            def loader():
                # Create a simple gradient image
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (index * 10) % 256
                img[:, :, 1] = np.arange(64).reshape(-1, 1)
                img[:, :, 2] = np.arange(64).reshape(1, -1)
                return img

            return loader

        return Frame(
            index=idx,
            timestamp=idx / 30.0,
            images={
                "camera0": LazyImage(
                    loader=make_image_loader(idx),
                    height=64,
                    width=64,
                    channels=3,
                ),
            },
            state=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            action=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], dtype=np.float32),
        )

    def frame_loader():
        for i in range(30):  # 1 second at 30 fps
            yield create_frame(i)

    return Episode(
        episode_id="ep_001",
        language_instruction="Pick up the red block",
        cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
        fps=30.0,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def sample_episodes(sample_episode: Episode) -> list[Episode]:
    """Create multiple sample episodes."""

    def create_episode(ep_idx: int) -> Episode:
        def create_frame(idx: int) -> Frame:
            def make_image_loader(index: int):
                def loader():
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    img[:, :, 0] = (index * 10) % 256
                    return img

                return loader

            return Frame(
                index=idx,
                timestamp=idx / 30.0,
                images={
                    "camera0": LazyImage(
                        loader=make_image_loader(idx),
                        height=64,
                        width=64,
                        channels=3,
                    ),
                },
                state=np.array([0.1] * 7, dtype=np.float32),
                action=np.array([0.01] * 7, dtype=np.float32),
            )

        def frame_loader():
            for i in range(20 + ep_idx * 5):  # Variable length episodes
                yield create_frame(i)

        return Episode(
            episode_id=f"ep_{ep_idx:03d}",
            language_instruction=f"Task {ep_idx % 2}",  # Two different tasks
            cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

    return [create_episode(i) for i in range(3)]


class TestLeRobotV3WriterConfig:
    """Tests for LeRobotV3WriterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LeRobotV3WriterConfig()
        assert config.fps == 30.0
        assert config.robot_type == "unknown"
        assert config.video_codec == "libx264"
        assert config.video_crf == 23
        assert config.chunks_size == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LeRobotV3WriterConfig(
            fps=60.0,
            robot_type="franka",
            video_crf=18,
            chunks_size=500,
            camera_name_mapping={"cam0": "observation.images.front"},
        )
        assert config.fps == 60.0
        assert config.robot_type == "franka"
        assert config.video_crf == 18
        assert config.chunks_size == 500
        assert config.camera_name_mapping == {"cam0": "observation.images.front"}

    def test_video_defaults_true(self):
        """Test that video mode is the default (backwards-compatible)."""
        config = LeRobotV3WriterConfig()
        assert config.video is True


class TestLeRobotV3Writer:
    """Tests for LeRobotV3Writer."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_single_episode(self, tmp_path: Path, sample_episode: Episode):
        """Test writing a single episode."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)  # Flush to write data

        # Check parquet file exists (chunked structure)
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        assert parquet_path.exists()

        # Check video file exists (chunked structure)
        video_path = output_dir / "videos" / "observation.images.camera0" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_dataset(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test writing full dataset with multiple episodes."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Check structure
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "data" / "chunk-000").exists()
        assert (output_dir / "videos").exists()

        # Check data parquet exists
        parquet_files = list((output_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(parquet_files) == 1

        # Check metadata content
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["fps"] == 30
        assert info["total_episodes"] == 3
        assert info["splits"]["train"] == "0:3"
        assert "data_path" in info
        assert "video_path" in info

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_episodes_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test episodes parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read episodes parquet
        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(episodes_path)
        episodes_data = table.to_pylist()

        assert len(episodes_data) == 3
        assert episodes_data[0]["episode_index"] == 0
        assert episodes_data[1]["episode_index"] == 1
        assert episodes_data[2]["episode_index"] == 2
        assert all("length" in ep for ep in episodes_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_tasks_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test tasks.parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read tasks.parquet
        tasks_path = output_dir / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_path)
        tasks_data = table.to_pylist()

        # Should have 2 unique tasks (Task 0 and Task 1)
        assert len(tasks_data) == 2
        assert any(t["task"] == "Task 0" for t in tasks_data)
        assert any(t["task"] == "Task 1" for t in tasks_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_camera_name_mapping(self, tmp_path: Path, sample_episode: Episode):
        """Test custom camera name mapping."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(
            fps=30.0,
            camera_name_mapping={"camera0": "observation.images.front_cam"},
        )
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Check video is in mapped directory
        video_path = output_dir / "videos" / "observation.images.front_cam" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_parquet_has_required_columns(self, tmp_path: Path, sample_episode: Episode):
        """Test parquet file has required columns."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Read parquet and check columns
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(parquet_path)
        columns = set(table.column_names)

        assert "episode_index" in columns
        assert "frame_index" in columns
        assert "index" in columns
        assert "task_index" in columns
        assert "timestamp" in columns
        assert "observation.state" in columns
        assert "action" in columns

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_default_camera_name_stripping(self, tmp_path: Path):
        """Test that _image and _rgb suffixes are stripped by default."""
        config = LeRobotV3WriterConfig()
        writer = LeRobotV3Writer(config)

        assert writer._map_camera_name("agentview_image") == "observation.images.agentview"
        assert writer._map_camera_name("front_rgb") == "observation.images.front"
        assert writer._map_camera_name("wrist") == "observation.images.wrist"

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_info_json_structure(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test info.json has correct LeRobot v3 structure."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        # Check required fields
        assert "codebase_version" in info
        assert "robot_type" in info
        assert "total_episodes" in info
        assert "total_frames" in info
        assert "total_tasks" in info
        assert "chunks_size" in info
        assert "fps" in info
        assert "splits" in info
        assert "data_path" in info
        assert "video_path" in info
        assert "features" in info

        # Check features have correct structure
        features = info["features"]
        assert "episode_index" in features
        assert "frame_index" in features
        assert "timestamp" in features
        assert "action" in features

        # Check video features have video_info
        for key, feat in features.items():
            if feat.get("dtype") == "video":
                assert "video_info" in feat
                assert "video.fps" in feat["video_info"]


class TestLeRobotV3WriterImageMode:
    """Tests for image mode (`LeRobotV3WriterConfig(video=False)`)."""

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_no_video_directory_created(self, tmp_path: Path, sample_episode: Episode):
        """Image mode never touches the video encoder or videos/ directory."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, video=False)
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        assert not (output_dir / "videos").exists()
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        assert parquet_path.exists()

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_frames_stored_inline_as_png(self, tmp_path: Path, sample_episode: Episode):
        """Camera columns hold PNG bytes decodable back to the source frame size."""
        import io

        import pyarrow.parquet as pq
        from PIL import Image

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, video=False)
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")
        rows = table.to_pylist()
        assert len(rows) == 30

        cam_col = "observation.images.camera0"
        expected = [f.images["camera0"].load() for f in sample_episode.frames()]
        for row, source in zip(rows, expected):
            assert cam_col in row
            assert row[cam_col]["path"] is None
            img = Image.open(io.BytesIO(row[cam_col]["bytes"]))
            assert img.format == "PNG"
            assert img.size == (64, 64)
            np.testing.assert_array_equal(np.asarray(img), source)

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_float_frames_in_0_255_range_are_not_rescaled(self, tmp_path: Path):
        """float32 frames already in 0-255 are clipped, not multiplied by 255."""
        import io

        import pyarrow.parquet as pq
        from PIL import Image

        source = np.full((8, 8, 3), 200.0, dtype=np.float32)

        def frame_loader():
            yield Frame(
                index=0,
                timestamp=0.0,
                images={
                    "camera0": LazyImage(
                        loader=lambda: source, height=8, width=8, channels=3
                    ),
                },
                state=np.zeros(3, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
            )

        episode = Episode(
            episode_id="ep_float",
            language_instruction="float frames",
            cameras={"camera0": CameraInfo(name="camera0", height=8, width=8)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

        output_dir = tmp_path / "output"
        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, video=False))
        writer.write_episode(episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")
        row = table.to_pylist()[0]
        decoded = np.asarray(Image.open(io.BytesIO(row["observation.images.camera0"]["bytes"])))
        np.testing.assert_array_equal(decoded, np.full((8, 8, 3), 200, dtype=np.uint8))

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_info_json_dtype_is_image(self, tmp_path: Path, sample_episodes: list[Episode]):
        """info.json declares image features with dtype 'image', not 'video'."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot", video=False)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        cam_feature = info["features"]["observation.images.camera0"]
        assert cam_feature["dtype"] == "image"
        assert "video_info" not in cam_feature
        assert cam_feature["fps"] == 30.0
        assert cam_feature["shape"] == [64, 64, 3]

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_info_json_video_path_is_null(self, tmp_path: Path, sample_episodes: list[Episode]):
        """info.json's video_path/video_files_size_in_mb are null in image
        mode, matching upstream LeRobot's use_videos=False convention
        (the keys stay present since v3 loaders require them to exist)."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, video=False)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert "video_path" in info
        assert info["video_path"] is None
        assert "video_files_size_in_mb" in info
        assert info["video_files_size_in_mb"] is None

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_episodes_parquet_has_no_video_pointers(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        """Image datasets carry no per-camera videos/ pointer columns."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, video=False)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(episodes_path)
        video_cols = [c for c in table.column_names if c.startswith("videos/")]
        assert video_cols == []

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_frame_count_matches_episode_length(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        """Total frame count and per-episode lengths are unaffected by image mode."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, video=False)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_frames"] == sum(20 + i * 5 for i in range(3))

        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_data = pq.read_table(episodes_path).to_pylist()
        assert episodes_data[0]["length"] == 20

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_stats_cover_state_and_action(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        """Normalization stats survive finalize's projected read of the data parquet."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, video=False))

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "stats.json") as f:
            stats = json.load(f)
        assert stats["observation.state"]["min"] == pytest.approx([0.1] * 7)
        assert stats["observation.state"]["max"] == pytest.approx([0.1] * 7)
        assert stats["action"]["mean"] == pytest.approx([0.01] * 7)

        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episode_row = pq.read_table(episodes_path).to_pylist()[0]
        assert episode_row["stats/observation.state/min"] == pytest.approx([0.1] * 7)
        assert episode_row["stats/observation.state/count"] == [20]

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_feature_shape_comes_from_decoded_frame(self, tmp_path: Path):
        """Shape is taken from the decoded array, not the LazyImage's declared dims."""
        source = np.zeros((12, 20, 3), dtype=np.uint8)

        def frame_loader():
            yield Frame(
                index=0,
                timestamp=0.0,
                images={
                    "camera0": LazyImage(
                        loader=lambda: source, height=48, width=64, channels=3
                    ),
                },
                state=np.zeros(3, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
            )

        episode = Episode(
            episode_id="ep_stale_dims",
            language_instruction="stale declared dims",
            cameras={"camera0": CameraInfo(name="camera0", height=48, width=64)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

        output_dir = tmp_path / "output"
        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, video=False))

        writer.write_dataset(iter([episode]), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["features"]["observation.images.camera0"]["shape"] == [12, 20, 3]

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_camera_missing_on_first_frame_still_written(self, tmp_path: Path):
        """A camera absent from frame 0 is not dropped from the data parquet.

        pyarrow derives the parquet schema from the first row alone, so a
        column that only shows up in later rows used to vanish silently while
        info.json still declared the feature.
        """
        import io

        import pyarrow.parquet as pq
        from PIL import Image

        def make_frame(idx: int) -> Frame:
            images = {}
            if idx > 0:
                images["camera0"] = LazyImage(
                    loader=lambda i=idx: np.full((8, 8, 3), i * 10, dtype=np.uint8),
                    height=8,
                    width=8,
                    channels=3,
                )
            return Frame(
                index=idx,
                timestamp=idx / 30.0,
                images=images,
                state=np.zeros(3, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
            )

        def frame_loader():
            for i in range(4):
                yield make_frame(i)

        episode = Episode(
            episode_id="ep_late_camera",
            language_instruction="camera appears late",
            cameras={"camera0": CameraInfo(name="camera0", height=8, width=8)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

        output_dir = tmp_path / "output"
        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, video=False))
        writer.write_dataset(iter([episode]), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            declared = set(json.load(f)["features"])
        rows = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet").to_pylist()

        cam_col = "observation.images.camera0"
        assert cam_col in declared
        assert declared <= set(rows[0])
        assert rows[0][cam_col] is None
        decoded = np.asarray(Image.open(io.BytesIO(rows[2][cam_col]["bytes"])))
        np.testing.assert_array_equal(decoded, np.full((8, 8, 3), 20, dtype=np.uint8))


class TestLeRobotV3WriterVideoModeUnchanged:
    """Regression: video mode (the default) is untouched by image mode support."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_video_mode_still_writes_mp4_and_no_inline_bytes(
        self, tmp_path: Path, sample_episode: Episode
    ):
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)  # video=True by default
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        video_path = (
            output_dir / "videos" / "observation.images.camera0" / "chunk-000" / "file-000.mp4"
        )
        assert video_path.exists()

        table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")
        assert "observation.images.camera0" not in table.column_names

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_video_mode_info_json_dtype_is_video(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        cam_feature = info["features"]["observation.images.camera0"]
        assert cam_feature["dtype"] == "video"
        assert "video_info" in cam_feature


class TestLeRobotV3WriterSparseColumns:
    """Columns whose first row is null must not break stats or feature inference."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_action_missing_on_first_frame_still_finalizes(self, tmp_path: Path):
        """An action absent from frame 0 must not fail stats computation.

        MCAP/RLDS readers emit ``action=None`` for frames with no action
        sample, so a column can legitimately be null at row 0 and populated
        later.
        """

        def frame_loader():
            for i in range(4):
                yield Frame(
                    index=i,
                    timestamp=i / 30.0,
                    images={},
                    state=np.full(3, float(i), dtype=np.float32),
                    action=None if i == 0 else np.full(3, float(i), dtype=np.float32),
                )

        episode = Episode(
            episode_id="ep_sparse_action",
            language_instruction="action appears late",
            cameras={},
            fps=30.0,
            _frame_loader=frame_loader,
        )

        output_dir = tmp_path / "output"
        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0))
        writer.write_dataset(iter([episode]), output_dir)

        with open(output_dir / "meta" / "stats.json") as f:
            stats = json.load(f)
        assert stats["action"]["count"] == [3]
        assert stats["action"]["min"] == [1.0, 1.0, 1.0]
        assert stats["action"]["max"] == [3.0, 3.0, 3.0]

    @pytest.mark.skipif(
        not _check_image_mode_dependencies_available(),
        reason="PIL or PyArrow not installed",
    )
    def test_parallel_inference_uses_first_non_null_row(self, tmp_path: Path):
        """Parallel-mode feature inference reads past a null first row.

        Workers write the data parquet themselves, so ``finalize`` infers
        features from the file. A camera missing on frame 0 leaves a null
        there; keying off row 0 alone declared it as float32.
        """
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        from forge.core.models import DatasetInfo

        def png(value: int) -> bytes:
            buf = io.BytesIO()
            Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(buf, format="PNG")
            return buf.getvalue()

        cam_col = "observation.images.camera0"
        rows = [
            {
                "episode_index": 0,
                "frame_index": i,
                "index": i,
                "task_index": 0,
                "timestamp": i / 30.0,
                "observation.state": None if i == 0 else [0.0, 0.0, 0.0],
                cam_col: None if i == 0 else {"bytes": png(i * 10), "path": None},
            }
            for i in range(3)
        ]
        data_dir = tmp_path / "output" / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), data_dir / "file-000.parquet")

        writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30.0, video=False))
        writer._total_frames = 3
        writer.finalize(
            tmp_path / "output",
            DatasetInfo(path=tmp_path, format="test", num_episodes=1, total_frames=3),
        )

        with open(tmp_path / "output" / "meta" / "info.json") as f:
            features = json.load(f)["features"]
        assert features[cam_col]["dtype"] == "image"
        assert features[cam_col]["shape"] == [8, 12, 3]
        assert features["observation.state"]["shape"] == [3]


JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]


@pytest.mark.skipif(
    not _check_image_mode_dependencies_available(),
    reason="PIL or PyArrow not installed",
)
class TestLeRobotV3WriterVectorNames:
    """State and action joint names reach info.json, and every episode must agree on them."""

    def _write(self, tmp_path: Path, episodes: list[Episode]) -> dict:
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot", video=False)
        LeRobotV3Writer(config).write_dataset(iter(episodes), tmp_path / "output")
        with open(tmp_path / "output" / "meta" / "info.json") as f:
            return json.load(f)["features"]

    def test_names_are_written(self, tmp_path: Path, sample_episodes: list[Episode]):
        for episode in sample_episodes:
            episode.state_names = list(JOINTS)
            episode.action_names = list(reversed(JOINTS))
        features = self._write(tmp_path, sample_episodes)
        assert features["observation.state"]["names"] == JOINTS
        assert features["action"]["names"] == list(reversed(JOINTS))

    def test_unnamed_episodes_write_null_names(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        features = self._write(tmp_path, sample_episodes)
        assert features["observation.state"]["names"] is None
        assert features["action"]["names"] is None

    def test_episodes_that_disagree_on_names_are_refused(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        for episode in sample_episodes:
            episode.state_names = list(JOINTS)
        sample_episodes[1].state_names = list(reversed(JOINTS))
        with pytest.raises(ConversionError, match="episode 'ep_001' lists observation.state names"):
            self._write(tmp_path, sample_episodes)

    def test_a_named_episode_after_an_unnamed_one_is_refused(
        self, tmp_path: Path, sample_episodes: list[Episode]
    ):
        sample_episodes[1].state_names = list(JOINTS)
        with pytest.raises(ConversionError, match="earlier episodes list None"):
            self._write(tmp_path, sample_episodes)
