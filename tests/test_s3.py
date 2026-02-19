from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest
from nd2 import ND2File, imread


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


@pytest.fixture()
def s3_nd2_url(small_nd2s: Path, s3_endpoint: str) -> Iterable[tuple[str, str]]:
    """Yield an S3 URL pointing to `single_nd2`, along with the endpoint URL."""
    if TYPE_CHECKING:
        import boto3
    else:
        boto3 = pytest.importorskip("boto3")

    bucket = f"nd2-test-{uuid4().hex[:12]}"
    client = boto3.client("s3", endpoint_url=s3_endpoint, region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    client.upload_file(str(small_nd2s), bucket, small_nd2s.name)
    yield f"s3://{bucket}/{small_nd2s.name}", s3_endpoint


def test_nd2file_reads_from_s3_url(
    s3_nd2_url: tuple[str, str], small_nd2s: Path
) -> None:
    url, endpoint = s3_nd2_url
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
