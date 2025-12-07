"""OME-Zarr export functionality for ND2 files.

This module provides functions to export ND2 files to OME-Zarr format
using yaozarrs for metadata generation and writing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nd2._ome import nd2_ome_metadata
from nd2._util import AXIS

try:
    from yaozarrs import v05
    from yaozarrs.write.v05 import write_bioformats2raw, write_image
except ImportError:
    raise ImportError(
        "yaozarrs is required for OME-Zarr export. "
        "Please pip install with either `nd2[ome-zarr]` or `nd2[ome-zarr-tensorstore]`"
    ) from None

if TYPE_CHECKING:
    from os import PathLike

    from typing_extensions import TypeAlias

    from nd2 import ND2File

    ZarrBackend: TypeAlias = Literal["zarr", "tensorstore", "auto"]


# OME-NGFF axis ordering
_OME_AXIS_ORDER = {
    AXIS.TIME: 0,
    AXIS.CHANNEL: 1,
    AXIS.Z: 2,
    AXIS.Y: 3,
    AXIS.X: 4,
    AXIS.RGB: 5,
}


def nd2_to_ome_zarr(
    nd2_file: ND2File,
    dest: str | PathLike,
    *,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None = "auto",
    shard_shape: tuple[int, ...] | None = None,
    backend: ZarrBackend = "auto",
    progress: bool = False,
    position: int | None = None,
    force_series: bool = False,
    version: Literal["0.5"] = "0.5",
) -> Path:
    """Export an ND2 file to OME-Zarr format.

    Creates a Zarr v3 store with OME-NGFF compliant metadata.
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
    backend : "zarr" | "tensorstore" | "auto"
        Backend library to use for writing arrays.
        - "tensorstore": Uses Google's tensorstore library
        - "zarr": Uses zarr-python
        - "auto": Tries to use tensorstore if installed, otherwise falls back
          to zarr-python. Raises ImportError if neither is available.
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
    version : "0.5"
        OME-NGFF specification version to use. Currently only "0.5" is
        supported. This parameter is reserved for future use.

    Returns
    -------
    Path
        Path to the created Zarr store.

    Raises
    ------
    ImportError
        If the required backend library is not installed.
    ValueError
        If the file contains unsupported data structures or invalid version.

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
    if version != "0.5":
        raise ValueError(
            f"Only version '0.5' is supported, got '{version}'. "
            "This parameter is reserved for future use."
        )

    dest_path = Path(dest)

    # Handle position axis specially
    nd2_sizes = dict(nd2_file.sizes)
    has_positions = AXIS.POSITION in nd2_sizes
    n_positions = nd2_sizes.pop(AXIS.POSITION, 1)

    # Check for unsupported RGB + channel combination
    has_rgb = AXIS.RGB in nd2_sizes
    has_channels = AXIS.CHANNEL in nd2_sizes
    if has_rgb and has_channels:
        raise ValueError(
            "OME-NGFF does not support files with both RGB samples and multiple "
            "optical channels. This ND2 file has both 'C' (channel) and 'S' (RGB) "
            "dimensions, which cannot be represented in the OME-Zarr format."
        )

    # For pure RGB files (no C axis), treat RGB samples as the channel axis
    is_rgb_image = has_rgb and not has_channels
    if is_rgb_image:
        rgb_size = nd2_sizes.pop(AXIS.RGB)
        nd2_sizes[AXIS.CHANNEL] = rgb_size

    # Validate position index
    if position is not None and position >= n_positions:
        raise IndexError(
            f"Position {position} out of range. File has {n_positions} positions."
        )

    if position is not None:
        positions_to_export = [position]
    else:
        positions_to_export = list(range(n_positions))

    # Get OME-Zarr axis order (excluding position)
    axes_order = sorted(nd2_sizes, key=lambda ax: _OME_AXIS_ORDER.get(ax, 99))

    if len(positions_to_export) == 1 and not force_series:
        # Single image - write directly to dest
        pos_idx = positions_to_export[0] if has_positions else None
        data = _get_position_data(
            nd2_file, pos_idx, axes_order, nd2_sizes, is_rgb_image
        )
        image_model = _build_image_model(
            nd2_file, axes_order, name=dest_path.stem, is_rgb=is_rgb_image
        )
        return write_image(  # type: ignore[no-any-return]
            dest_path,
            image_model,
            [data],
            chunks=chunk_shape,
            shards=shard_shape,
            writer=backend,
            progress=progress,
        )
    else:
        # Multiple positions - use bioformats2raw layout
        images: dict[str, tuple[v05.Image, list[Any]]] = {}
        for pos_idx in positions_to_export:
            data = _get_position_data(
                nd2_file, pos_idx, axes_order, nd2_sizes, is_rgb_image
            )
            image_model = _build_image_model(
                nd2_file, axes_order, name=str(pos_idx), is_rgb=is_rgb_image
            )
            images[str(pos_idx)] = (image_model, [data])

        # Generate OME-XML if possible
        ome_xml: str | None = None
        try:
            ome_metadata = nd2_ome_metadata(nd2_file, include_unstructured=False)
            ome_xml = ome_metadata.to_xml()
        except NotImplementedError as e:
            warnings.warn(f"Could not generate OME-XML metadata: {e}. ", stacklevel=2)

        return write_bioformats2raw(  # type: ignore[no-any-return]
            dest_path,
            images,
            ome_xml=ome_xml,
            chunks=chunk_shape,
            shards=shard_shape,
            writer=backend,
            progress=progress,
        )


# ######################## ND2-Specific Helpers ################################


def _build_image_model(
    nd2_file: ND2File,
    axes_order: list[str],
    name: str | None = None,
    is_rgb: bool = False,
) -> v05.Image:
    """Build yaozarrs Image model from nd2 metadata."""
    axes = _create_axes(axes_order)
    scales = _get_scale_values(nd2_file, axes_order)

    multiscale = v05.Multiscale(
        name=name,
        axes=axes,
        datasets=[
            v05.Dataset(
                path="0",
                coordinateTransformations=[v05.ScaleTransformation(scale=scales)],
            )
        ],
    )

    # Create omero metadata for RGB images
    omero = None
    if is_rgb:
        omero = v05.Omero(
            channels=[
                v05.OmeroChannel(color="FF0000", label="Red", active=True),
                v05.OmeroChannel(color="00FF00", label="Green", active=True),
                v05.OmeroChannel(color="0000FF", label="Blue", active=True),
            ],
            rdefs=v05.OmeroRenderingDefs(model="color"),
        )

    return v05.Image(multiscales=[multiscale], omero=omero)


def _create_axes(
    axes_order: list[str],
) -> list[v05.TimeAxis | v05.ChannelAxis | v05.SpaceAxis]:
    """Map nd2 axes to yaozarrs axis objects."""
    axis_map = {
        AXIS.TIME: v05.TimeAxis(name="t", unit="millisecond"),
        AXIS.CHANNEL: v05.ChannelAxis(name="c"),
        AXIS.Z: v05.SpaceAxis(name="z", unit="micrometer"),
        AXIS.Y: v05.SpaceAxis(name="y", unit="micrometer"),
        AXIS.X: v05.SpaceAxis(name="x", unit="micrometer"),
        AXIS.RGB: v05.ChannelAxis(name="s"),
    }
    return [axis_map[ax] for ax in axes_order if ax in axis_map]


def _get_scale_values(nd2_file: ND2File, axes_order: list[str]) -> list[float]:
    """Get scale values for coordinate transformations."""
    voxel = nd2_file.voxel_size()

    # Get time interval if present
    time_interval_ms = 1.0
    for loop in nd2_file.experiment:
        if loop.type == "TimeLoop":
            params = loop.parameters
            if params.periodDiff and params.periodDiff.avg:
                time_interval_ms = params.periodDiff.avg
            else:
                time_interval_ms = params.periodMs
            break
        elif loop.type == "NETimeLoop":
            if loop.parameters.periods:
                p = loop.parameters.periods[0]
                if p.periodDiff and p.periodDiff.avg:
                    time_interval_ms = p.periodDiff.avg
                else:
                    time_interval_ms = p.periodMs
            break

    scale_map = {
        AXIS.TIME: time_interval_ms,
        AXIS.CHANNEL: 1.0,
        AXIS.Z: voxel.z,
        AXIS.Y: voxel.y,
        AXIS.X: voxel.x,
        AXIS.RGB: 1.0,
    }
    return [scale_map[ax] for ax in axes_order if ax in scale_map]


def _get_axis_permutation(
    nd2_sizes: dict[str, int], target_axes: list[str]
) -> tuple[int, ...]:
    """Get permutation indices for np.transpose to reorder ND2 axes to target order."""
    nd2_axes = list(nd2_sizes.keys())
    return tuple(nd2_axes.index(ax) for ax in target_axes if ax in nd2_axes)


def _get_position_data(
    nd2_file: ND2File,
    pos_idx: int | None,
    axes_order: list[str],
    nd2_sizes: dict[str, int],
    is_rgb: bool = False,
) -> Any:
    """Get data for a position, handling axis permutation."""
    original_sizes = dict(nd2_file.sizes)
    has_positions = AXIS.POSITION in original_sizes

    # For RGB files, we treat S as C for axis ordering purposes
    if is_rgb and AXIS.RGB in original_sizes:
        rgb_size = original_sizes.pop(AXIS.RGB)
        original_sizes[AXIS.CHANNEL] = rgb_size

    if has_positions and pos_idx is not None:
        data = nd2_file.asarray(position=pos_idx)
        pos_dim_idx = list(nd2_file.sizes).index(AXIS.POSITION)
        data = np.squeeze(data, axis=pos_dim_idx)
        sizes_no_pos = {k: v for k, v in original_sizes.items() if k != AXIS.POSITION}
        perm = _get_axis_permutation(sizes_no_pos, axes_order)
    else:
        data = nd2_file.to_dask()
        perm = _get_axis_permutation(nd2_sizes, axes_order)

    if perm != tuple(range(data.ndim)):
        data = data.transpose(perm)

    return data
