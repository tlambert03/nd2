"""Extract frames from corrupted ND2 files."""

from ast import parse
import sys
import argparse
from typing import Sequence

from pathlib import Path

import numpy as np

import nd2
from ._parse._chunk_decode import rescue_nd2


def existing_path(path: str) -> Path:
    """Check if the provided path exists."""
    _path = Path(path)
    if not _path.exists():
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist.")
    return _path


def parse_frame_shape(shape: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers into a tuple of integers."""
    try:
        return tuple(map(int, shape.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid frame shape: '{shape}'")


def _parse_args(argv: Sequence[str] = ()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump all salvageable frames in an nd2 file to a folder."
    )
    parser.add_argument(
        "path",
        type=existing_path,
        help="Path to ND2 file or directory containing ND2 files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for extracted frames. By default, will "
        "create a directory with the same name as the input file.",
    )
    parser.add_argument(
        "--frame-shape",
        type=parse_frame_shape,  # Use the custom type function
        default=None,
        help="Shape of the frames in the ND2 file as a comma-separated list of "
        "integers. If not provided, the an attempt will be made to read shape from the "
        "file.",
    )
    parser.add_argument(
        "--dtype",
        type=np.dtype,
        default=None,
        help="Data type of the frames in the ND2 file. If not provided, the an attempt "
        "will be made to read the data type from the file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    return parser.parse_args(argv or sys.argv[1:])


def main(argv: Sequence[str] = ()) -> None:
    """Index ND2 files and print the results as a table."""
    try:
        import tifffile as tf
    except ImportError:
        raise ImportError(
            "The tifffile package is required to convert nd2 to tiff files. "
            "Please install it with `pip install tifffile` or `pip install nd2[tiff]`."
        ) from None

    args = _parse_args(argv)
    fpath: Path = args.path

    frame_shape = args.frame_shape
    dtype = args.dtype
    if not (frame_shape and dtype):
        try:
            with nd2.ND2File(fpath) as f:
                if args.verbose:
                    print(f"Reading frame shape and dtype from {fpath}")                
                if not frame_shape:
                    frame_shape = f._frame_shape
                if not dtype:
                    dtype = f.dtype
                print(f.attributes)
                print(f.experiment)
                print(f._frame_shape)
        except Exception as e:
            if args.verbose:
                print(
                    f"Failed to read frame shape and dtype from file {e}. "
                    "You may pass them manually using --frame-shape and --dtype."
                )
            frame_shape = frame_shape or ()
            dtype = dtype or np.dtype("uint16")

    output: Path = args.output or fpath.with_suffix(".frames")
    output.mkdir(exist_ok=True)
    fstem = fpath.stem
    for i, frame in enumerate(
        rescue_nd2(fpath, frame_shape=frame_shape, dtype=dtype, verbose=args.verbose)
    ):
        dest = output / f"{fstem}_{i:06d}.tiff"
        tf.imwrite(dest, frame, imagej=True)
        if args.verbose:
            print(f"Saved frame {i} to {dest}")


if __name__ == "__main__":
    main()
