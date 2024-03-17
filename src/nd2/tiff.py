"""Functions for converting nd2 to tiff files."""

from __future__ import annotations

import warnings
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from nd2._ome import nd2_ome_metadata
from nd2._util import AXIS
from nd2.nd2file import ND2File

try:
    import tifffile as tf
except ImportError as e:
    raise ImportError(
        "The tifffile package is required to convert nd2 to tiff files. "
        "Please install it with `pip install tifffile` or `pip install nd2[tiff]`."
    ) from e


try:
    from tqdm import tqdm as _pbar

except ImportError:

    class _pbar:  # type: ignore
        def __init__(
            self, *_: Any, desc: str = "", total: int | None = None, **__: Any
        ) -> None:
            self.desc = desc
            self.total = total or "?"
            self.n = 0
            print("hint: `pip install tqdm` for a better progress bar. ")

        def set_description(self, desc: str) -> None:
            self.desc = desc

        def update(self, n: int = 1) -> None:
            self.n += n
            end = "\n" if self.n == self.total else "\r"
            print(f"  Writing frame {self.n} of {self.total}: {self.desc}", end=end)
            if self.n == self.total:
                print("  Done!")

        def close(self) -> None: ...


if TYPE_CHECKING:
    import numpy as np
    import ome_types

    from .nd2file import ND2File


def nd2_to_tiff(
    source: str | PathLike | ND2File,
    dest: str | PathLike,
    *,
    include_unstructured_metadata: bool = True,
    progress: bool = False,
    on_frame: Callable[[int, int, dict[str, int]], None] | None = None,
    modify_ome: Callable[[ome_types.OME], None] | None = None,
) -> None:
    """Export an ND2 file to an (OME)-TIFF file.

    To include OME-XML metadata, use extension `.ome.tif` or `.ome.tiff`.

    https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html

    Parameters
    ----------
    source : str | PathLike | ND2File
        The ND2 file path or an open ND2File object.
    dest : str  | PathLike
        The destination TIFF file.
    include_unstructured_metadata :  bool
        Whether to include unstructured metadata in the OME-XML. This includes all of
        the metadata that we can find in the ND2 file in the StructuredAnnotations
        section of the OME-XML (as mapping of metadata chunk name to JSON-encoded
        string). By default `True`.
    progress : bool
        Whether to display progress bar.  If `True` and `tqdm` is installed, it will
        be used. Otherwise, a simple text counter will be printed to the console.
        By default `False`.
    on_frame : Callable[[int, int, dict[str, int]], None] | None
        A function to call after each frame is written. The function should accept
        three arguments: the current frame number, the total number of frames, and
        a dictionary of the current frame's indices (e.g. `{"T": 0, "Z": 1}`)
        (Useful for integrating custom progress bars or logging.)
    modify_ome : Callable[[ome_types.OME], None]
        A function to modify the OME metadata before writing it to the file.
        Accepts an `ome_types.OME` object and should modify it in place.
        (reminder: OME-XML is only written if the file extension is `.ome.tif` or
        `.ome.tiff`)
    """
    dest_path = Path(dest).expanduser().resolve()
    output_ome = ".ome." in dest_path.name

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
        # map of axis_name -> size
        sizes = dict(nd2f.sizes)

        # pop the number of positions from the sizes.
        # The OME data model does best with 5D data, so we'll write multi-5D series
        n_positions = sizes.pop(AXIS.POSITION, 1)

        # join axis names as a string, and get shape of the data without positions
        axes, shape = zip(*sizes.items())
        # U (Unknown) -> Q : other (OME)
        metadata = {"axes": "".join(axes).upper().replace(AXIS.UNKNOWN, "Q")}

        # Create OME-XML
        ome_xml: bytes | None = None
        if output_ome:
            if nd2f.is_legacy:
                warnings.warn(
                    "Cannot write OME metadata for legacy nd2 files."
                    "Please use a different file extension to avoid confusion",
                    stacklevel=2,
                )
            else:
                # get the OME metadata object from the ND2File
                ome = nd2_ome_metadata(
                    nd2f,
                    include_unstructured=include_unstructured_metadata,
                    tiff_file_name=dest_path.name,
                )
                if modify_ome:
                    # allow user to modify the OME metadata if they want
                    modify_ome(ome)
                ome_xml = ome.to_xml(exclude_unset=True).encode("utf-8")

        # total number of frames we will write
        tot = nd2f._frame_count
        # create a progress bar if requested
        pbar = _pbar(total=tot, desc=f"Exporting {nd2f.path}") if progress else None

        # `p_groups` will be a map of {position index -> [(frame_number, f_index) ...]}
        # where frame_number is passed to read_frame
        # and f_index is a map of axis name to index (e.g. {"T": 0, "Z": 1})
        # positions are grouped together so we can write them to the tiff file in order
        p_groups: defaultdict[int, list[tuple[int, dict[str, int]]]] = defaultdict(list)
        for f_num, f_index in enumerate(nd2f.loop_indices):
            p_groups[f_index.get(AXIS.POSITION, 0)].append((f_num, f_index))

        # create a function to iterate over all frames, updating pbar if requested
        def position_iter(p: int) -> Iterator[np.ndarray]:
            """Iterator over frames for a given position."""
            for f_num, f_index in p_groups[p]:
                # call on_frame callback if provided
                if on_frame is not None:
                    on_frame(f_num, tot, f_index)

                # yield the frame and update the progress bar
                yield nd2f.read_frame(f_num)
                if pbar is not None:
                    pbar.set_description(repr(f_index))
                    pbar.update()

        # if we have ome_xml, we tell tifffile not to worry about it (ome=False)
        tf_ome = False if ome_xml else None
        # Write the tiff file
        pixelsize = nd2f.voxel_size().x
        photometric = tf.PHOTOMETRIC.RGB if nd2f.is_rgb else tf.PHOTOMETRIC.MINISBLACK
        with tf.TiffWriter(dest_path, bigtiff=True, ome=tf_ome) as tif:
            for p in range(n_positions):
                tif.write(
                    iter(position_iter(p)),
                    shape=shape,
                    dtype=nd2f.dtype,
                    resolution=(1 / pixelsize, 1 / pixelsize),
                    resolutionunit=tf.TIFF.RESUNIT.MICROMETER,
                    photometric=photometric,
                    metadata=metadata,
                    description=ome_xml,
                )

        if pbar is not None:
            pbar.close()

    finally:
        # close the nd2 file if we opened it
        if close_when_done:
            nd2f.close()
