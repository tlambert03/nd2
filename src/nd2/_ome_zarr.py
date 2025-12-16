"""OME-Zarr export functionality for ND2 files.

This module provides functions to export ND2 files to OME-Zarr format
using yaozarrs for metadata generation and writing.
"""

from __future__ import annotations

import json
import re
import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nd2 import __version__
from nd2._ome import nd2_ome_metadata
from nd2._util import AXIS

try:
    from yaozarrs import v05
    from yaozarrs.write.v05 import (
        Bf2RawBuilder,
        LabelsBuilder,
        PlateBuilder,
        write_image,
    )
except ImportError:
    raise ImportError(
        "yaozarrs is required for OME-Zarr export. "
        "Please pip install with either `nd2[ome-zarr]` or `nd2[ome-zarr-tensorstore]`"
    ) from None

if TYPE_CHECKING:
    from os import PathLike

    from typing_extensions import TypeAlias

    from nd2 import ND2File
    from nd2._binary import BinaryLayer, BinaryLayers

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


WELL_PATTERN = re.compile(r"^([A-Z]+)(\d+)$")


def _json_default(obj: Any) -> Any:
    """JSON encoder for non-serializable types in nd2 metadata."""
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    return str(obj)  # pragma: no cover


def _add_nd2_attributes(zarr_path: Path, nd2_file: ND2File) -> None:
    """Add nd2 metadata to the root zarr.json attributes.

    Modifies the zarr.json file to include nd2-specific metadata in the
    attributes object alongside the "ome" key.

    Parameters
    ----------
    zarr_path : Path
        Path to the root zarr directory.
    nd2_file : ND2File
        The ND2 file to extract metadata from.
    """
    zarr_json_path = zarr_path / "zarr.json"
    if not zarr_json_path.exists():
        return  # pragma: no cover

    zarr_json = json.loads(zarr_json_path.read_text())
    nd2_attrs: dict[str, Any] = {"version": __version__}
    with suppress(Exception):
        nd2_attrs["unstructured_metadata"] = nd2_file.unstructured_metadata()
    with suppress(Exception):
        nd2_attrs["custom_data"] = nd2_file.custom_data
    zarr_json.setdefault("attributes", {})["nd2"] = nd2_attrs

    zarr_json_path.write_text(json.dumps(zarr_json, indent=2, default=_json_default))


def _detect_wellplate(
    nd2_file: ND2File,
) -> dict[tuple[str, str, str], int] | None:
    """Detect if file is from a multi-well plate and map (row, col, fov) to position.

    Returns
    -------
    dict[tuple[str, str, str], int] | None
        Mapping of (row, col, fov) tuples to position indices if this is a wellplate,
        None otherwise. Example: {("A", "1", "0"): 0, ("A", "4", "0"): 1, ...}

    Raises
    ------
    NotImplementedError
        If multiple fields of view per well are detected. Please open an issue
        at https://github.com/tlambert03/nd2/issues with the file.
    """
    if AXIS.POSITION not in nd2_file.sizes:
        return None

    well_positions: dict[tuple[str, str, str], int] = {}
    seen_wells: set[tuple[str, str]] = set()

    for pos_idx in range(nd2_file.sizes[AXIS.POSITION]):
        try:
            meta = nd2_file.frame_metadata(pos_idx)
            # Type guard: ensure we have FrameMetadata, not dict
            if isinstance(meta, dict) or not meta.channels:
                return None

            position = meta.channels[0].position
            if not position or not position.name:
                return None

            if not (match := WELL_PATTERN.match(position.name)):
                return None

            row, col = match.groups()
            # Strip leading zeros from column to match yaozarrs plate column naming
            col = col.lstrip("0") or "0"

            # Check for multiple fields of view per well
            if (row, col) in seen_wells:  # pragma: no cover
                raise NotImplementedError(
                    f"Multiple fields of view per well detected (well {row}{col}). "
                    "This feature is not yet implemented. Please open an issue at "
                    "https://github.com/tlambert03/nd2/issues with your file."
                )

            seen_wells.add((row, col))
            fov = "0"
            well_positions[(row, col, fov)] = pos_idx

        except (AttributeError, IndexError, KeyError):
            return None

    return well_positions if well_positions else None


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
    include_labels: bool = True,
    version: Literal["0.5"] = "0.5",
    overwrite: bool = False,
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
    include_labels : bool
        If True (default), export binary masks as OME-Zarr labels.
        Binary masks from the ND2 file will be written to a "labels"
        subdirectory within the image group. Each binary layer becomes
        a separate label with its own name. Has no effect if the file
        contains no binary data.
    version : "0.5"
        OME-NGFF specification version to use. Currently only "0.5" is
        supported. This parameter is reserved for future use.
    overwrite: bool
        If True, overwrite the destination directory if it already exists.

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
        positions_to_export: list[int] = [position]
    else:
        positions_to_export = list(range(n_positions))

    # Get OME-Zarr axis order (excluding position)
    axes_order = sorted(nd2_sizes, key=lambda ax: _OME_AXIS_ORDER.get(ax, 99))

    # Detect if this is a multi-well plate
    well_positions = _detect_wellplate(nd2_file) if position is None else None

    if well_positions is not None:
        # Multi-well plate - use Plate layout
        plate_builder = PlateBuilder(
            dest_path,
            writer=backend,
            chunks=chunk_shape,
            shards=shard_shape,
            overwrite=overwrite,
        )

        # Group positions by (row, col) -> {fov: pos_idx}
        wells: dict[tuple[str, str], dict[str, int]] = {}
        for (row, col, fov), pos_idx in well_positions.items():
            wells.setdefault((row, col), {})[fov] = pos_idx

        # Write each well with its fields
        for (row, col), fields in wells.items():
            fields_data: dict[str, tuple[v05.Image, Any]] = {}
            for fov, pos_idx in fields.items():
                data = _get_position_data(
                    nd2_file, pos_idx, axes_order, nd2_sizes, is_rgb_image
                )
                image_model = _build_image_model(
                    nd2_file, axes_order, name=f"{row}{col}", is_rgb=is_rgb_image
                )
                fields_data[fov] = (image_model, data)

            plate_builder.write_well(
                row=row, col=col, images=fields_data, progress=progress
            )

            # Write labels for each field in this well
            if include_labels and (binary_data := nd2_file.binary_data):
                well_path = dest_path / row / col
                for fov, pos_idx in fields.items():
                    field_path = well_path / fov
                    _write_labels(
                        field_path / "labels",
                        binary_data,
                        pos_idx,
                        axes_order,
                        nd2_file,
                        backend,
                        chunk_shape,
                        shard_shape,
                        progress,
                        overwrite,
                    )

        result_path = Path(plate_builder.root_path)
        _add_nd2_attributes(result_path, nd2_file)
        return result_path
    elif len(positions_to_export) == 1 and not force_series:
        # Single image - write directly to dest
        p_idx = positions_to_export[0] if has_positions else None
        data = _get_position_data(nd2_file, p_idx, axes_order, nd2_sizes, is_rgb_image)
        image_model = _build_image_model(
            nd2_file, axes_order, name=dest_path.stem, is_rgb=is_rgb_image
        )
        root = write_image(
            dest_path,
            image_model,
            data,
            chunks=chunk_shape,
            shards=shard_shape,
            writer=backend,
            overwrite=overwrite,
            progress=progress,
        )

        # Write binary masks as labels if present
        if include_labels and nd2_file.binary_data:
            _write_labels(
                dest_path / "labels",
                nd2_file.binary_data,
                p_idx,
                axes_order,
                nd2_file,
                backend,
                chunk_shape,
                shard_shape,
                progress,
                overwrite,
            )

        result_path = Path(root)
        _add_nd2_attributes(result_path, nd2_file)
        return result_path
    else:
        # Multiple positions - use bioformats2raw layout
        # Generate OME-XML if possible
        ome_xml: str | None = None
        try:
            ome_metadata = nd2_ome_metadata(nd2_file, include_unstructured=False)
            ome_xml = ome_metadata.to_xml()
        except NotImplementedError as e:
            warnings.warn(f"Could not generate OME-XML metadata: {e}. ", stacklevel=2)

        # Use builder pattern for efficient incremental writes
        builder = Bf2RawBuilder(
            dest_path,
            ome_xml=ome_xml,
            writer=backend,
            chunks=chunk_shape,
            shards=shard_shape,
            overwrite=overwrite,
        )

        for pos_idx in positions_to_export:
            image_model = _build_image_model(
                nd2_file, axes_order, name=str(pos_idx), is_rgb=is_rgb_image
            )
            data = _get_position_data(
                nd2_file, pos_idx, axes_order, nd2_sizes, is_rgb_image
            )
            builder.write_image(str(pos_idx), image_model, data, progress=progress)

            # Write labels for this position if present
            if include_labels and nd2_file.binary_data:
                position_path = dest_path / str(pos_idx)
                _write_labels(
                    position_path / "labels",
                    nd2_file.binary_data,
                    pos_idx,
                    axes_order,
                    nd2_file,
                    backend,
                    chunk_shape,
                    shard_shape,
                    progress,
                    overwrite,
                )

        result_path = Path(builder.root_path)
        _add_nd2_attributes(result_path, nd2_file)
        return result_path


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
    """Get data for a position as a lazy dask array, handling axis permutation.

    Always returns a dask array to keep data out of RAM until yaozarrs writes it.
    """
    original_sizes = dict(nd2_file.sizes)
    has_positions = AXIS.POSITION in original_sizes

    # For RGB files, we treat S as C for axis ordering purposes
    if is_rgb and AXIS.RGB in original_sizes:
        rgb_size = original_sizes.pop(AXIS.RGB)
        original_sizes[AXIS.CHANNEL] = rgb_size

    # Always use dask to keep data lazy until write time
    data = nd2_file.to_dask()

    if has_positions and pos_idx is not None:
        # Slice the dask array to select the position (stays lazy)
        pos_dim_idx = list(nd2_file.sizes).index(AXIS.POSITION)
        slices: list[int | slice] = [slice(None)] * data.ndim
        slices[pos_dim_idx] = pos_idx
        data = data[tuple(slices)]
        sizes_no_pos = {k: v for k, v in original_sizes.items() if k != AXIS.POSITION}
        perm = _get_axis_permutation(sizes_no_pos, axes_order)
    else:
        perm = _get_axis_permutation(nd2_sizes, axes_order)

    if perm != tuple(range(data.ndim)):
        data = data.transpose(perm)

    return data


def _build_label_image(
    axes_order: list[str],
    scales: list[float],
    name: str,
) -> v05.LabelImage:
    """Build yaozarrs LabelImage model for a binary layer.

    Parameters
    ----------
    axes_order : list[str]
        Ordered list of axis names (e.g., ['T', 'Z', 'Y', 'X']).
    scales : list[float]
        Scale values for each axis.
    name : str
        Name of the label.

    Returns
    -------
    v05.LabelImage
        LabelImage model for use with LabelsBuilder.
    """
    axes = _create_axes(axes_order)

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

    return v05.LabelImage(
        multiscales=[multiscale],
        image_label=v05.ImageLabel(),
    )


def _get_label_data(
    binary_layer: BinaryLayer,
    nd2_file: ND2File,
    pos_idx: int | None,
    axes_order: list[str],
) -> np.ndarray:
    """Get binary layer data with proper axis ordering for OME-Zarr.

    Parameters
    ----------
    binary_layer : BinaryLayer
        Binary layer from nd2 file.
    nd2_file : ND2File
        The ND2 file to get axis information from.
    pos_idx : int | None
        Position index if exporting a single position, None otherwise.
    axes_order : list[str]
        Target axis order (e.g., ['T', 'Z', 'Y', 'X']).

    Returns
    -------
    np.ndarray
        Binary mask data with axes reordered to match OME-Zarr format.
    """
    # Get the full array (e.g., (T, P, Z, Y, X))
    data = binary_layer.asarray()
    if data is None:  # pragma: no cover
        warnings.warn(
            f"No data found for binary layer '{binary_layer.name}'.", stacklevel=2
        )
        # Return empty array if no data
        return np.array([])

    # Binary layers follow the ND2 coordinate structure but exclude channel
    # Build list of axes present in binary layer from nd2 file structure
    nd2_axes = [ax for ax in nd2_file.sizes if ax != AXIS.CHANNEL]

    # If we're exporting a single position, select it from the data
    if pos_idx is not None and AXIS.POSITION in nd2_axes:
        p_axis_idx = nd2_axes.index(AXIS.POSITION)
        data = np.take(data, pos_idx, axis=p_axis_idx)
        # Remove position from axes list
        nd2_axes = [ax for ax in nd2_axes if ax != AXIS.POSITION]

    # Now reorder axes to match the target axes_order
    # Filter to only axes that exist in both lists
    target_axes = [ax for ax in axes_order if ax in nd2_axes]

    # Get permutation from current order to target order
    if target_axes and len(target_axes) == data.ndim:
        perm = tuple(nd2_axes.index(ax) for ax in target_axes)
        if perm != tuple(range(data.ndim)):
            data = data.transpose(perm)

    return data


def _write_labels(
    labels_path: Path,
    binary_data: BinaryLayers,
    pos_idx: int | None,
    axes_order: list[str],
    nd2_file: ND2File,
    backend: ZarrBackend,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None,
    shard_shape: tuple[int, ...] | None,
    progress: bool,
    overwrite: bool,
) -> None:
    """Write binary masks as OME-Zarr labels.

    Parameters
    ----------
    labels_path : Path
        Path to the labels directory.
    binary_data : BinaryLayers
        Binary layers from nd2 file.
    pos_idx : int | None
        Position index if exporting a single position, None otherwise.
    axes_order : list[str]
        Target axis order for the labels.
    nd2_file : ND2File
        The ND2 file being exported.
    backend : ZarrBackend
        Backend to use for writing.
    chunk_shape : tuple[int, ...] | "auto" | None
        Chunk shape for the labels.
    shard_shape : tuple[int, ...] | None
        Shard shape for the labels.
    progress : bool
        Whether to show progress bar.
    overwrite : bool
        Whether to overwrite existing labels.
    """
    # Filter axes to exclude channel (labels don't have channel dimension)
    label_axes_order = [ax for ax in axes_order if ax != AXIS.CHANNEL]
    scales = _get_scale_values(nd2_file, label_axes_order)

    # Create labels builder
    labels_builder = LabelsBuilder(
        labels_path,
        writer=backend,
        chunks=chunk_shape,
        shards=shard_shape,
        overwrite=overwrite,
    )

    # Write each binary layer as a separate label
    for i, layer in enumerate(binary_data):
        # Sanitize layer name for use as path (replace spaces, special chars)
        safe_name = layer.name.replace(" ", "_").replace("(", "").replace(")", "")
        safe_name = f"label_{i}_{safe_name}" if safe_name else f"label_{i}"

        # Get the data with proper axis ordering
        label_data = _get_label_data(layer, nd2_file, pos_idx, label_axes_order)

        if label_data.size == 0:
            continue

        # Build label metadata
        label_image = _build_label_image(label_axes_order, scales, layer.name)

        # Write the label
        labels_builder.write_label(
            safe_name,
            label_image,
            label_data,
            progress=progress,
        )
