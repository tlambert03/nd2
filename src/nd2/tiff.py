"""Functions for converting nd2 to tiff files."""

from os import PathLike
from typing import TYPE_CHECKING, Any

from nd2.nd2file import ND2File

try:
    import tifffile as tf
except ImportError as e:
    raise ImportError(
        "The tifffile package is required to convert nd2 to tiff files. "
        "Please install it with `pip install tifffile` or `pip install nd2[tiff]`."
    ) from e

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


if TYPE_CHECKING:
    from .nd2file import ND2File  # noqa: TCH004


def nd2_to_tiff(
    source: str | PathLike | ND2File,
    dest: str | PathLike,
    flush_every: int = 100,
    progress: bool = False,
):
    """Export an ND2 file to a TIFF file.

    Parameters
    ----------
    source : str | PathLike | ND2File
        The source ND2 file.
    dest : str  | PathLike
        The destination TIFF file.
    flush_every : int
        The number of frames to write before flushing the memory-mapped array to disk.
    progress : bool
        Whether to display a progress bar.
    """
    close = False
    if isinstance(source, (str, PathLike)):
        from .nd2file import ND2File

        source = ND2File(source)
        close = True

    try:
        dims, shape = zip(*source.sizes.items())
        axes = "".join(dims).upper().replace("U", "Q")  # Q : other (OME)
        metadata: dict[str, Any] = {"axes": axes}

        # write empty file to disk
        tf.imwrite(
            dest,
            shape=shape,
            dtype=source.dtype,
            bigtiff=True,
            metadata=metadata,
            ome=True,
        )

        # memory-mapped NumPy array of image data stored in TIFF file.
        mmap = tf.memmap(dest, dtype=source.dtype)
        # This line is important, as tifffile.memmap appears to lose singleton dims
        mmap.shape = shape

        tot = source._frame_count
        for frame_num, frame_index in tqdm(
            enumerate(source.loop_indices), total=tot, desc=f"Exporting {source.path}"
        ):
            index = tuple(v for k, v in frame_index.items() if k in dims)
            frame_data = source.read_frame(frame_num)

            # WRITE DATA TO DISK
            mmap[index] = frame_data
            if frame_num % flush_every == 0:
                mmap.flush()

        # Write ome metadata
        tf.tiffcomment(dest, source.ome_metadata().to_xml().encode("ascii", "ignore"))

    finally:
        if close:
            source.close()
