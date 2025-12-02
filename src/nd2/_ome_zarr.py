"""OME-Zarr export functionality for ND2 files.

This module provides functions to export ND2 files to OME-Zarr format
using yaozarrs for metadata generation and either zarr-python or
tensorstore for array writing.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from typing import TYPE_CHECKING, Any, Literal

from nd2._util import AXIS

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    from typing_extensions import TypeAlias

    from nd2 import ND2File

    # Type alias for backend selection
    ZarrBackend: TypeAlias = Literal["zarr", "tensorstore"]


def _get_ome_axes_order(nd2_sizes: dict[str, int]) -> list[str]:
    """Determine OME-Zarr axes order from ND2 sizes.

    OME-Zarr requires axes in order: time, channel, spatial (z, y, x).
    ND2 files may have axes in different orders.

    Parameters
    ----------
    nd2_sizes : dict[str, int]
        The sizes dict from ND2File.sizes

    Returns
    -------
    list[str]
        Ordered list of axis names in OME-Zarr order.
    """
    axes = []
    # Time first (if present)
    if AXIS.TIME in nd2_sizes:
        axes.append(AXIS.TIME)
    # Channel second (if present)
    if AXIS.CHANNEL in nd2_sizes:
        axes.append(AXIS.CHANNEL)
    # Spatial axes last, in z, y, x order
    if AXIS.Z in nd2_sizes:
        axes.append(AXIS.Z)
    if AXIS.Y in nd2_sizes:
        axes.append(AXIS.Y)
    if AXIS.X in nd2_sizes:
        axes.append(AXIS.X)
    # RGB/sample dimension comes after spatial (if present)
    if AXIS.RGB in nd2_sizes:
        axes.append(AXIS.RGB)
    return axes


def _get_axis_permutation(
    nd2_sizes: dict[str, int], target_axes: list[str]
) -> tuple[int, ...]:
    """Get permutation indices to reorder ND2 axes to target order.

    Parameters
    ----------
    nd2_sizes : dict[str, int]
        The sizes dict from ND2File.sizes (in ND2 axis order)
    target_axes : list[str]
        Target axis order (OME-Zarr order)

    Returns
    -------
    tuple[int, ...]
        Permutation indices for np.transpose
    """
    nd2_axes = list(nd2_sizes.keys())
    return tuple(nd2_axes.index(ax) for ax in target_axes if ax in nd2_axes)


def _create_yaozarrs_axes(
    nd2_file: ND2File, axes_order: list[str]
) -> list[dict[str, Any]]:
    """Create yaozarrs-compatible axis definitions.

    Parameters
    ----------
    nd2_file : ND2File
        The open ND2 file
    axes_order : list[str]
        Axes in OME-Zarr order

    Returns
    -------
    list[dict[str, Any]]
        List of axis definitions compatible with yaozarrs v05 models
    """
    from yaozarrs.v05 import ChannelAxis, SpaceAxis, TimeAxis

    nd2_file.voxel_size()
    axes: list[Any] = []

    for ax in axes_order:
        if ax == AXIS.TIME:
            axes.append(TimeAxis(name="t", unit="millisecond"))
        elif ax == AXIS.CHANNEL:
            axes.append(ChannelAxis(name="c"))
        elif ax == AXIS.Z:
            axes.append(SpaceAxis(name="z", unit="micrometer"))
        elif ax == AXIS.Y:
            axes.append(SpaceAxis(name="y", unit="micrometer"))
        elif ax == AXIS.X:
            axes.append(SpaceAxis(name="x", unit="micrometer"))
        elif ax == AXIS.RGB:
            # RGB is typically handled as a channel-type axis
            axes.append(ChannelAxis(name="s"))

    return axes


def _get_scale_values(nd2_file: ND2File, axes_order: list[str]) -> list[float]:
    """Get scale values for coordinate transformations.

    Parameters
    ----------
    nd2_file : ND2File
        The open ND2 file
    axes_order : list[str]
        Axes in OME-Zarr order

    Returns
    -------
    list[float]
        Scale values for each axis
    """
    voxel = nd2_file.voxel_size()

    # Get time interval if present
    time_interval_ms = 1.0  # Default
    for loop in nd2_file.experiment:
        if loop.type == "TimeLoop":
            params = loop.parameters
            if params.periodDiff and params.periodDiff.avg:
                time_interval_ms = params.periodDiff.avg
            else:
                time_interval_ms = params.periodMs
            break
        elif loop.type == "NETimeLoop":
            # For NETimeLoop, use average from first period
            if loop.parameters.periods:
                p = loop.parameters.periods[0]
                if p.periodDiff and p.periodDiff.avg:
                    time_interval_ms = p.periodDiff.avg
                else:
                    time_interval_ms = p.periodMs
            break

    scales = []
    for ax in axes_order:
        if ax == AXIS.TIME:
            scales.append(time_interval_ms)
        elif ax == AXIS.CHANNEL:
            scales.append(1.0)  # Channel has no physical scale
        elif ax == AXIS.Z:
            scales.append(voxel.z)
        elif ax == AXIS.Y:
            scales.append(voxel.y)
        elif ax == AXIS.X:
            scales.append(voxel.x)
        elif ax == AXIS.RGB:
            scales.append(1.0)  # RGB has no physical scale

    return scales


def _create_multiscale_metadata(
    nd2_file: ND2File,
    dataset_paths: list[str],
    axes_order: list[str],
    name: str | None = None,
) -> dict[str, Any]:
    """Create OME-Zarr multiscale metadata.

    Parameters
    ----------
    nd2_file : ND2File
        The open ND2 file
    dataset_paths : list[str]
        Relative paths to each resolution level
    axes_order : list[str]
        Axes in OME-Zarr order
    name : str, optional
        Name for the multiscale

    Returns
    -------
    dict[str, Any]
        Multiscale metadata dict
    """
    from yaozarrs.v05 import Dataset, Image, Multiscale, ScaleTransformation

    axes = _create_yaozarrs_axes(nd2_file, axes_order)
    scale_values = _get_scale_values(nd2_file, axes_order)

    # Create datasets (currently only one resolution level)
    datasets = []
    for i, path in enumerate(dataset_paths):
        # For multiple resolution levels, scale would be multiplied
        # For now we only have one level
        downscale_factor = 2**i if i > 0 else 1
        current_scale = [s * downscale_factor for s in scale_values]

        datasets.append(
            Dataset(
                path=path,
                coordinateTransformations=[ScaleTransformation(scale=current_scale)],
            )
        )

    multiscale = Multiscale(
        name=name or nd2_file._path.stem if hasattr(nd2_file, "_path") else None,
        axes=axes,
        datasets=datasets,
    )

    image = Image(multiscales=[multiscale])
    return image.model_dump(mode="json", exclude_none=True)  # type: ignore[no-any-return]


def _write_zarr_json(path: Path, metadata: dict[str, Any]) -> None:
    """Write zarr.json file with OME metadata.

    Parameters
    ----------
    path : Path
        Path to the zarr directory
    metadata : dict[str, Any]
        OME metadata from yaozarrs
    """
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": metadata},
    }
    (path / "zarr.json").write_text(json.dumps(zarr_json, indent=2))


def _ensure_chunks(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...] | None,
    dtype_itemsize: int,
) -> tuple[int, ...]:
    """Determine chunk shape for array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape
    chunk_shape : tuple[int, ...] | None
        User-specified chunk shape, or None for auto
    dtype_itemsize : int
        Size of dtype in bytes

    Returns
    -------
    tuple[int, ...]
        Chunk shape to use
    """
    if chunk_shape is not None:
        # Ensure chunks don't exceed shape
        return tuple(min(c, s) for c, s in zip(chunk_shape, shape))

    # Auto-determine chunks targeting ~16-64 MB chunks
    target_bytes = 32 * 1024 * 1024  # 32 MB
    target_bytes // dtype_itemsize

    # Start with shape, reduce from the end (spatial dims)
    chunks = list(shape)

    # Always chunk time and channel dimensions to 1 for efficient slicing
    if len(chunks) >= 5:  # TCZYX
        chunks[0] = 1  # T
        chunks[1] = 1  # C
    elif len(chunks) >= 4:  # CZYX or TZYX
        chunks[0] = 1

    # Limit spatial dimensions
    for i in range(len(chunks) - 1, -1, -1):
        if i >= len(chunks) - 3:  # Spatial dims
            chunks[i] = min(chunks[i], 512)

    return tuple(chunks)


def _write_array_zarr(
    path: Path,
    data: Any,
    chunks: tuple[int, ...],
    shard_shape: tuple[int, ...] | None = None,
    dimension_names: list[str] | None = None,
    progress: bool = False,
) -> None:
    """Write array using zarr-python backend.

    Parameters
    ----------
    path : Path
        Path to write array
    data : array-like
        Data to write (numpy or dask array)
    chunks : tuple[int, ...]
        Chunk shape
    shard_shape : tuple[int, ...] | None
        Shard shape for sharded storage, or None for regular chunks
    dimension_names : list[str] | None
        Names for each dimension
    progress : bool
        Whether to show progress
    """
    import zarr
    from zarr.codecs import BloscCodec

    # Determine if data is dask
    is_dask = hasattr(data, "compute")

    # Create store
    store = zarr.storage.LocalStore(str(path))

    # Configure compressor
    compressor = BloscCodec(cname="zstd", clevel=3)

    # Create array with optional sharding
    arr = zarr.create_array(
        store,
        shape=data.shape,
        chunks=chunks,
        shards=shard_shape,
        dtype=data.dtype,
        compressors=compressor,
        dimension_names=dimension_names,
    )

    # Write data
    if is_dask:
        import dask.array as da

        if progress:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                da.store(data, arr, lock=False)
        else:
            da.store(data, arr, lock=False)
    else:
        arr[:] = data


def _write_array_tensorstore(
    path: Path,
    data: Any,
    chunks: tuple[int, ...],
    shard_shape: tuple[int, ...] | None = None,
    dimension_names: list[str] | None = None,
    progress: bool = False,
) -> None:
    """Write array using tensorstore backend.

    Parameters
    ----------
    path : Path
        Path to write array
    data : array-like
        Data to write (numpy or dask array)
    chunks : tuple[int, ...]
        Chunk shape
    shard_shape : tuple[int, ...] | None
        Shard shape for sharded storage
    dimension_names : list[str] | None
        Names for each dimension
    progress : bool
        Whether to show progress
    """
    import tensorstore as ts

    # Determine if data is dask
    is_dask = hasattr(data, "compute")

    # Build spec
    spec: dict[str, Any] = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(data.shape),
            "data_type": str(data.dtype),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(chunks)},
            },
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {
                    "name": "blosc",
                    "configuration": {"cname": "zstd", "clevel": 3},
                },
            ],
        },
        "create": True,
        "delete_existing": True,
    }

    if dimension_names:
        spec["metadata"]["dimension_names"] = dimension_names

    if shard_shape is not None:
        spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"] = list(
            shard_shape
        )
        spec["metadata"]["codecs"] = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(chunks),
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "blosc",
                            "configuration": {"cname": "zstd", "clevel": 3},
                        },
                    ],
                },
            }
        ]

    store = ts.open(spec).result()

    # Write data
    if is_dask:
        # For dask, we need to compute and write

        if progress:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                computed = data.compute()
        else:
            computed = data.compute()
        store[:].write(computed).result()
    else:
        store[:].write(data).result()


def nd2_to_ome_zarr(
    nd2_file: ND2File,
    dest: str | PathLike,
    *,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None = "auto",
    shard_shape: tuple[int, ...] | None = None,
    backend: ZarrBackend = "zarr",
    progress: bool = False,
    position: int | None = None,
    force_series: bool = False,
) -> Path:
    """Export an ND2 file to OME-Zarr format.

    Creates a Zarr v3 store with OME-NGFF 0.5 compliant metadata.
    The output uses yaozarrs for metadata generation and can use either
    zarr-python or tensorstore for array writing.

    Parameters
    ----------
    nd2_file : ND2File
        An open ND2File object to export.
    dest : str | PathLike
        Destination path for the Zarr store. Will be created as a directory.
    chunk_shape : tuple[int, ...] | "auto" | None
        Shape of chunks for the output array. If "auto" (default), determines
        optimal chunking based on data size. If None, uses a single chunk.
    shard_shape : tuple[int, ...] | None
        Shape of shards for sharded storage. If provided, enables Zarr v3
        sharding where each shard contains multiple chunks. Useful for
        cloud storage to reduce number of objects.
    backend : "zarr" | "tensorstore"
        Backend library to use for writing arrays.
        - "zarr": Uses zarr-python (default)
        - "tensorstore": Uses Google's tensorstore library
    progress : bool
        Whether to display a progress bar during writing.
    position : int | None
        If the ND2 file contains multiple positions (XY stage positions),
        export only this position index. If None, exports all positions
        as separate groups within the store.
    force_series : bool
        If True, use bioformats2raw layout even for single position files.
        This creates a store with OME/ directory and series metadata,
        with the image in a "0/" subdirectory. Default is False.

    Returns
    -------
    Path
        Path to the created Zarr store.

    Raises
    ------
    ImportError
        If the required backend library is not installed.
    ValueError
        If the file contains unsupported data structures.

    Examples
    --------
    Basic export:

    >>> import nd2
    >>> with nd2.ND2File("experiment.nd2") as f:
    ...     f.to_ome_zarr("experiment.zarr")

    Export with specific chunking and sharding:

    >>> with nd2.ND2File("experiment.nd2") as f:
    ...     f.to_ome_zarr(
    ...         "experiment.zarr",
    ...         chunk_shape=(1, 1, 64, 256, 256),
    ...         shard_shape=(1, 1, 256, 1024, 1024),
    ...     )

    Export using tensorstore backend:

    >>> with nd2.ND2File("experiment.nd2") as f:
    ...     f.to_ome_zarr("experiment.zarr", backend="tensorstore")
    """
    from pathlib import Path

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Determine axes order for OME-Zarr
    nd2_sizes = dict(nd2_file.sizes)

    # Handle position axis specially
    has_positions = AXIS.POSITION in nd2_sizes
    n_positions = nd2_sizes.pop(AXIS.POSITION, 1)

    if position is not None:
        if position >= n_positions:
            raise IndexError(
                f"Position {position} out of range. File has {n_positions} positions."
            )
        positions_to_export = [position]
    else:
        positions_to_export = list(range(n_positions))

    # Get OME-Zarr axis order (excluding position)
    axes_order = _get_ome_axes_order(nd2_sizes)
    permutation = _get_axis_permutation(nd2_sizes, axes_order)

    # Dimension names for zarr array
    dim_names = []
    for ax in axes_order:
        if ax == AXIS.TIME:
            dim_names.append("t")
        elif ax == AXIS.CHANNEL:
            dim_names.append("c")
        elif ax == AXIS.Z:
            dim_names.append("z")
        elif ax == AXIS.Y:
            dim_names.append("y")
        elif ax == AXIS.X:
            dim_names.append("x")
        elif ax == AXIS.RGB:
            dim_names.append("s")

    # Create dataset paths
    dataset_paths = ["0"]  # Currently single resolution

    # Prepare metadata
    metadata = _create_multiscale_metadata(
        nd2_file, dataset_paths, axes_order, name=dest_path.stem
    )

    # Select backend write function
    if backend == "zarr":
        if not importlib.util.find_spec("zarr"):
            raise ImportError(
                "zarr-python is required for the 'zarr' backend. "
                "Install with: pip install zarr>=3"
            )
        write_array = _write_array_zarr
    elif backend == "tensorstore":
        if not importlib.util.find_spec("tensorstore"):
            raise ImportError(
                "tensorstore is required for the 'tensorstore' backend. "
                "Install with: pip install tensorstore"
            )
        write_array = _write_array_tensorstore
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Export each position
    if len(positions_to_export) == 1 and not force_series:
        # Single position (or single position selected from multi-position file)
        # Write directly to dest
        _write_zarr_json(dest_path, metadata)

        # Get data for the position
        if has_positions:
            # Extract single position from multi-position file
            import numpy as np

            pos_idx = positions_to_export[0]
            data = nd2_file.asarray(position=pos_idx)
            # Squeeze out position dimension
            original_sizes = dict(nd2_file.sizes)
            pos_dim_idx = list(original_sizes.keys()).index(AXIS.POSITION)
            data = np.squeeze(data, axis=pos_dim_idx)
            # Get permutation without position axis
            sizes_no_pos = {
                k: v for k, v in original_sizes.items() if k != AXIS.POSITION
            }
            perm = _get_axis_permutation(sizes_no_pos, axes_order)
            if perm != tuple(range(data.ndim)):
                data = data.transpose(perm)
        else:
            # No positions, use dask array
            data = nd2_file.to_dask()
            if permutation != tuple(range(data.ndim)):
                data = data.transpose(permutation)

        # Determine chunks
        if chunk_shape == "auto":
            chunks = _ensure_chunks(data.shape, None, data.dtype.itemsize)
        elif chunk_shape is None:
            chunks = data.shape
        else:
            chunks = _ensure_chunks(data.shape, chunk_shape, data.dtype.itemsize)

        # Write array
        array_path = dest_path / "0"
        write_array(
            array_path,
            data,
            chunks,
            shard_shape=shard_shape,
            dimension_names=dim_names,
            progress=progress,
        )
    else:
        # Multiple positions - use bioformats2raw layout
        # Structure:
        #   root.zarr/
        #   ├── zarr.json          # bioformats2raw.layout metadata
        #   ├── OME/
        #   │   ├── zarr.json      # series metadata
        #   │   └── METADATA.ome.xml
        #   ├── 0/                 # First position (OME-Zarr Image)
        #   ├── 1/                 # Second position
        #   └── ...
        from yaozarrs.v05 import Bf2Raw, Series

        from nd2._ome import nd2_ome_metadata

        # Write root zarr.json with bioformats2raw.layout
        bf2raw = Bf2Raw(bioformats2raw_layout=3)
        root_zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": bf2raw.model_dump(mode="json", exclude_none=True)},
        }
        (dest_path / "zarr.json").write_text(json.dumps(root_zarr_json, indent=2))

        # Create OME directory with series metadata and METADATA.ome.xml
        ome_path = dest_path / "OME"
        ome_path.mkdir(parents=True, exist_ok=True)

        # Write OME/zarr.json with series list
        series_list = [str(i) for i in positions_to_export]
        series = Series(series=series_list)
        ome_zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": series.model_dump(mode="json", exclude_none=True)},
        }
        (ome_path / "zarr.json").write_text(json.dumps(ome_zarr_json, indent=2))

        # Generate and write METADATA.ome.xml
        ome_metadata = nd2_ome_metadata(nd2_file, include_unstructured=False)
        (ome_path / "METADATA.ome.xml").write_text(ome_metadata.to_xml())

        for pos_idx in positions_to_export:
            pos_name = str(pos_idx)
            pos_path = dest_path / pos_name

            # Get data for this position
            data = nd2_file.asarray(position=pos_idx)

            # Remove position dimension (asarray keeps it with size 1)
            original_sizes = dict(nd2_file.sizes)
            if AXIS.POSITION in original_sizes:
                # Find and squeeze out position dimension
                pos_dim_idx = list(original_sizes.keys()).index(AXIS.POSITION)
                data = data.squeeze(axis=pos_dim_idx)
                # Rebuild permutation without position axis
                sizes_no_pos = {
                    k: v for k, v in original_sizes.items() if k != AXIS.POSITION
                }
                perm = _get_axis_permutation(sizes_no_pos, axes_order)
            else:
                perm = permutation

            if perm != tuple(range(data.ndim)):
                data = data.transpose(perm)

            # Write position group metadata (each position is an OME-Zarr Image)
            pos_metadata = _create_multiscale_metadata(
                nd2_file, ["0"], axes_order, name=pos_name
            )
            pos_path.mkdir(parents=True, exist_ok=True)
            _write_zarr_json(pos_path, pos_metadata)

            # Determine chunks
            if chunk_shape == "auto":
                chunks = _ensure_chunks(data.shape, None, data.dtype.itemsize)
            elif chunk_shape is None:
                chunks = data.shape
            else:
                chunks = _ensure_chunks(data.shape, chunk_shape, data.dtype.itemsize)

            # Write array
            array_path = pos_path / "0"
            write_array(
                array_path,
                data,
                chunks,
                shard_shape=shard_shape,
                dimension_names=dim_names,
                progress=progress,
            )

    return dest_path
