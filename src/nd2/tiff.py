"""Functions for converting nd2 to tiff files."""

from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from nd2._ome import nd2_ome_metadata
from nd2.nd2file import ND2File

try:
    import tifffile as tf
except ImportError as e:
    raise ImportError(
        "The tifffile package is required to convert nd2 to tiff files. "
        "Please install it with `pip install tifffile` or `pip install nd2[tiff]`."
    ) from e

try:
    from tqdm import tqdm as _progress
except ImportError:

    def _progress(
        iterator: Iterable, *, total: int | None = None, **kwargs: Any
    ) -> Any:
        """Simple progress output if tqdm is unavailable."""
        print(kwargs.get("desc", ""))
        for i in iterator:
            print(f"  Writing frame {i + 1} of {total or '?'}", end="\r")
            yield i
        print()


if TYPE_CHECKING:
    import numpy as np
    import ome_types

    from .nd2file import ND2File


def nd2_to_tiff(
    source: str | PathLike | ND2File,
    dest: str | PathLike,
    progress: bool = True,
    on_frame: Callable[[int, int], None] | None = None,
    modify_ome: Callable[[ome_types.OME], None] | None = None,
) -> None:
    """Export an ND2 file to an (OME)-TIFF file.

    To include OME-XML metadata, use extension `.ome.tif` or `.ome.tiff`.

    Parameters
    ----------
    source : str | PathLike | ND2File
        The ND2 file path or an open ND2File object.
    dest : str  | PathLike
        The destination TIFF file.
    progress : bool
        Whether to display progress bar.  If `True` and `tqdm` is installed, it will
        be used. Otherwise, a simple text counter will be printed to the console.
    on_frame : Callable[[int, int], None]
        A function to call after each frame is written. The function should accept
        two arguments: the current frame number, and the total number of frames.
        (Useful for integrating custom progress bars or logging.)
    modify_ome : Callable[[ome_types.OME], None]
        A function to modify the OME metadata before writing it to the file.
        Accepts an `ome_types.OME` object and should modify it in place.
        (reminder: OME-XML is only written if the file extension is `.ome.tif` or
        `.ome.tiff`)
    """
    dest_path = Path(dest).expanduser().resolve()
    use_ome = dest_path.name.lower().endswith((".ome.tif", ".ome.tiff"))

    # normalize source to an open ND2File, and remember if we opened it
    close_when_done = False
    if isinstance(source, (str, PathLike)):
        from .nd2file import ND2File

        nd2f = ND2File(source)
        close_when_done = True
    else:
        nd2f = source
        if close_when_done := nd2f.closed:
            nd2f.open()

    try:
        # get shape and axes
        dims, shape = zip(*nd2f.sizes.items())
        # U (Unknown) -> Q : other (OME)
        # P (Position) -> R : tile (OME)
        # if "P" in dims:
        # raise NotImplementedError("Positions not written yet")
        axes = "".join(dims).upper().replace("U", "Q").replace("P", "R")
        metadata: dict[str, Any] = {"axes": axes}

        # create an iterator with progress bar if requested
        nframes = nd2f._frame_count
        indices: Iterable[int] = range(nframes)
        if progress:
            indices = _progress(indices, total=nframes, desc=f"Exporting {nd2f.path}")

        def dataiter() -> Iterator[np.ndarray]:
            for frame_num in indices:
                yield nd2f.read_frame(frame_num)
                # call on_frame callback if provided
                if on_frame is not None:
                    on_frame(frame_num, nframes)

        # Create OME-XML
        ome_xml: bytes | None = None
        if use_ome:
            if nd2f.is_legacy:
                warnings.warn(
                    "Cannot write OME metadata for legacy nd2 files."
                    "Please use a different file extension to avoid confusion",
                    stacklevel=2,
                )
            else:
                ome = nd2_ome_metadata(nd2f, tiff_file_name=dest_path.name)
                if modify_ome:
                    modify_ome(ome)
                # note, Christoph suggests encode("ascii")... but that changes µm to m
                # that could be addressed in ome_types, by serializing to um?
                ome_xml = ome.to_xml().encode("utf-8")

        # if we have ome_xml, we tell tifffile not to worry about it (ome=False)
        tf_ome = False if ome_xml else None
        # Write the tiff file
        pixelsize = nd2f.voxel_size().x
        with tf.TiffWriter(dest_path, bigtiff=True, ome=tf_ome) as tif:
            tif.write(
                iter(dataiter()),
                shape=shape,
                dtype=nd2f.dtype,
                resolution=(1 / pixelsize, 1 / pixelsize),
                resolutionunit=tf.TIFF.RESUNIT.MICROMETER,
                photometric=tf.TIFF.PHOTOMETRIC.MINISBLACK,
                metadata=metadata,
                description=ome_xml,
            )
    finally:
        # close the nd2 file if we opened it
        if close_when_done:
            nd2f.close()
