from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from nd2 import ND2File, imread

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from mypy_boto3_s3.service_resource import Bucket


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterable[str]:
    if TYPE_CHECKING:
        from moto.server import ThreadedMotoServer
    else:
        ThreadedMotoServer = pytest.importorskip("moto.server").ThreadedMotoServer
    server = ThreadedMotoServer(port=0)
    server.start()
    _, port = server.get_host_and_port()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


@pytest.fixture(scope="session")
def s3_bucket(s3_endpoint: str) -> Bucket:
    if TYPE_CHECKING:
        import boto3
    else:
        boto3 = pytest.importorskip("boto3")
    s3 = boto3.resource("s3", endpoint_url=s3_endpoint, region_name="us-east-1")
    return s3.create_bucket(Bucket="nd2-test")


@pytest.fixture()
def any_s3_nd2_url(small_nd2s: Path, s3_bucket: Bucket) -> Iterable[tuple[str, str]]:
    """Yield an S3 URL pointing to any nd2, along with the endpoint URL."""
    s3_bucket.upload_file(str(small_nd2s), small_nd2s.name)
    endpoint = s3_bucket.meta.client.meta.endpoint_url
    yield f"s3://{s3_bucket.name}/{small_nd2s.name}", endpoint


@pytest.fixture()
def single_s3_nd2_url(single_nd2: Path, s3_bucket: Bucket) -> Iterable[tuple[str, str]]:
    """Yield an S3 URL pointing to `single_nd2`, along with the endpoint URL."""
    s3_bucket.upload_file(str(single_nd2), single_nd2.name)
    endpoint = s3_bucket.meta.client.meta.endpoint_url
    yield f"s3://{s3_bucket.name}/{single_nd2.name}", endpoint


def test_nd2file_reads_from_s3_url(
    any_s3_nd2_url: tuple[str, str], small_nd2s: Path
) -> None:
    url, endpoint = any_s3_nd2_url
    storage_options = {
        "client_kwargs": {"endpoint_url": endpoint},
        # necessary for moto's S3 implementation
        # to avoid checksum validation errors on multipart uploads
        "config_kwargs": {"response_checksum_validation": "when_required"},
    }
    remote_nd = ND2File(url, storage_options=storage_options)
    local_nd = ND2File(small_nd2s)
    with remote_nd, local_nd:
        assert remote_nd.path == url
        assert remote_nd.shape == local_nd.shape
        assert remote_nd.metadata == local_nd.metadata
        assert remote_nd.attributes == local_nd.attributes
        assert remote_nd.text_info == local_nd.text_info
        assert remote_nd.experiment == local_nd.experiment
        assert remote_nd.frame_metadata(0) == local_nd.frame_metadata(0)
        assert remote_nd.events() == local_nd.events()
        assert remote_nd.ome_metadata() == local_nd.ome_metadata()
        np.testing.assert_array_equal(remote_nd.read_frame(0), local_nd.read_frame(0))
        if local_nd.binary_data is not None:
            assert remote_nd.binary_data is not None
            for rb, lb in zip(
                remote_nd.binary_data, local_nd.binary_data, strict=False
            ):
                np.testing.assert_array_equal(rb.asarray(), lb.asarray())
        for rr, lr in zip(remote_nd.rois.items(), local_nd.rois.items(), strict=False):
            assert rr == lr

    full_local_read = imread(small_nd2s)
    full_remote_read = imread(url, storage_options=storage_options)
    np.testing.assert_array_equal(full_remote_read, full_local_read)


def test_nd2file_reads_from_fsspec_obj(
    single_s3_nd2_url: tuple[str, str], single_nd2: Path
) -> None:
    """Test that the user can construct their own fsspec file-like for ND2File."""
    if TYPE_CHECKING:
        import fsspec
    else:
        fsspec = pytest.importorskip("fsspec")

    url, endpoint = single_s3_nd2_url
    storage_options = {
        "client_kwargs": {"endpoint_url": endpoint},
        "config_kwargs": {"response_checksum_validation": "when_required"},
    }
    fs = fsspec.filesystem("s3", **storage_options)
    with fs.open(url, "rb") as fs_fh:
        remote_nd = ND2File(fs_fh, storage_options=storage_options)
        local_nd = ND2File(single_nd2)
        with remote_nd, local_nd:
            assert remote_nd.path == url
            assert remote_nd.shape == local_nd.shape
            assert remote_nd.metadata == local_nd.metadata
            assert remote_nd.attributes == local_nd.attributes
            assert remote_nd.text_info == local_nd.text_info
            assert remote_nd.experiment == local_nd.experiment
            assert remote_nd.frame_metadata(0) == local_nd.frame_metadata(0)
            assert remote_nd.events() == local_nd.events()
            assert remote_nd.ome_metadata() == local_nd.ome_metadata()
            np.testing.assert_array_equal(
                remote_nd.read_frame(0), local_nd.read_frame(0)
            )
