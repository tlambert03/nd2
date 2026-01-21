"""Tests for remote/fsspec file access functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nd2._readers.protocol import _is_remote_path, _open_file

DATA = Path(__file__).parent / "data"


class TestIsRemotePath:
    """Tests for _is_remote_path() URL detection."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Remote URLs - should return True
            ("http://example.com/file.nd2", True),
            ("https://example.com/file.nd2", True),
            ("s3://bucket/file.nd2", True),
            ("gs://bucket/file.nd2", True),
            ("az://container/file.nd2", True),
            ("abfs://container/file.nd2", True),
            ("smb://server/share/file.nd2", True),
            # Local paths - should return False
            ("/path/to/file.nd2", False),
            ("./relative/path.nd2", False),
            ("file.nd2", False),
            (Path("/path/to/file.nd2"), False),
            (Path("relative/path.nd2"), False),
            # Edge cases
            ("", False),
            ("http", False),  # No :// suffix
            ("s3", False),
            # Non-string types
            (123, False),
            (None, False),
        ],
    )
    def test_remote_path_detection(self, path, expected):
        """Test that remote URLs are correctly identified."""
        assert _is_remote_path(path) == expected


class TestOpenFile:
    """Tests for _open_file() function."""

    def test_open_local_file(self, any_nd2):
        """Test opening a local file."""
        fh = _open_file(any_nd2)
        try:
            assert hasattr(fh, "read")
            assert hasattr(fh, "seek")
            # Should be able to read some bytes
            data = fh.read(4)
            assert len(data) == 4
        finally:
            fh.close()

    def test_open_local_path_object(self, any_nd2):
        """Test opening a local file using Path object."""
        fh = _open_file(Path(any_nd2))
        try:
            assert hasattr(fh, "read")
            data = fh.read(4)
            assert len(data) == 4
        finally:
            fh.close()

    def test_open_remote_without_fsspec(self):
        """Test that opening remote URL without fsspec raises ImportError."""
        with patch.dict("sys.modules", {"fsspec": None}):
            # Force reimport to pick up the patched module
            with pytest.raises(ImportError, match="fsspec is required"):
                _open_file("https://example.com/file.nd2")

    def test_open_remote_with_fsspec(self):
        """Test opening remote URL with mocked fsspec."""
        mock_file = MagicMock()
        mock_file.read.return_value = b"\x00\x00\x00\x00"
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_file

        mock_fsspec = MagicMock()
        mock_fsspec.open.return_value = mock_opener

        with patch.dict("sys.modules", {"fsspec": mock_fsspec}):
            # We need to reload to use the mock
            from nd2._readers import protocol

            # Call the function
            protocol._open_file("https://example.com/file.nd2")

            # Verify fsspec.open was called correctly
            mock_fsspec.open.assert_called_once()
            call_args = mock_fsspec.open.call_args
            assert call_args[0][0] == "https://example.com/file.nd2"
            assert call_args[1]["mode"] == "rb"
            assert "block_size" in call_args[1]


class TestND2FileRemote:
    """Integration tests for ND2File with remote URLs."""

    def test_is_remote_property_local(self, any_nd2):
        """Test that is_remote returns False for local files."""
        import nd2

        with nd2.ND2File(any_nd2) as f:
            assert f.is_remote is False

    def test_read_frames_local(self, any_nd2):
        """Test read_frames() method on local files."""
        import nd2

        with nd2.ND2File(any_nd2) as f:
            # Read a subset of frames
            n_frames = min(3, f.attributes.sequenceCount)
            if n_frames > 0:
                indices = list(range(n_frames))
                frames = f.read_frames(indices)
                assert isinstance(frames, np.ndarray)
                assert frames.shape[0] == n_frames

    def test_read_frames_all(self, any_nd2):
        """Test read_frames() with no indices reads all frames."""
        import nd2

        with nd2.ND2File(any_nd2) as f:
            seq_count = f.attributes.sequenceCount
            if seq_count > 0 and seq_count <= 10:  # Only test small files
                frames = f.read_frames()
                assert isinstance(frames, np.ndarray)
                assert frames.shape[0] == seq_count


class TestReadFramesParallel:
    """Tests for parallel frame reading."""

    def test_parallel_read_local_uses_sequential(self, any_nd2):
        """Test that local files use sequential reading (via mmap)."""
        import nd2

        with nd2.ND2File(any_nd2) as f:
            # Local files should have mmap, so parallel read falls back to sequential
            if f.attributes.sequenceCount > 0:
                frames = f.read_frames([0], max_workers=4)
                assert isinstance(frames, np.ndarray)
                assert frames.shape[0] == 1

    def test_parallel_read_with_mock_remote(self, any_nd2):
        """Test parallel reading behavior with mocked remote file."""
        import nd2
        from nd2._readers._modern.modern_reader import ModernReader

        # Read a local file first to get valid frame data
        with nd2.ND2File(any_nd2) as f:
            if f.attributes.sequenceCount == 0:
                pytest.skip("No frames in test file")

            # Get the reader and simulate remote by clearing mmap
            if isinstance(f._rdr, ModernReader):
                original_mmap = f._rdr._mmap
                f._rdr._mmap = None  # Simulate remote (no mmap)
                f._rdr._is_remote = True

                try:
                    # This should use the parallel reading path
                    frames = f._rdr.read_frames_parallel([0], max_workers=2)
                    assert isinstance(frames, list)
                    assert len(frames) == 1
                    assert isinstance(frames[0], np.ndarray)
                finally:
                    f._rdr._mmap = original_mmap
                    f._rdr._is_remote = False
