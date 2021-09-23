from types import TracebackType
from typing import (
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    overload,
)

import numpy as np
from typing_extensions import Literal

from . import structures

class Attributes(NamedTuple):
    bitsPerComponentInMemory: int  # bits allocated to hold each component
    bitsPerComponentSignificant: int  # bits effectively used by each component
    componentCount: int  # number of components in a pixel
    heightPx: int  # height of the image
    pixelDataType: str  # underlying data type "unsigned" or "float"
    sequenceCount: int  # number of image frames in the file
    widthBytes: int  # number of bytes from the beginning of one line to the next one
    widthPx: int  # width of the image
    compressionLevel: Optional[int]  # if compression is used the level of compression
    compressionType: Optional[str]  # type of compression: "lossless" or "lossy"
    tileHeightPx: Optional[int]  # suggested tile height if saved as tiled
    tileWidthPx: Optional[int]  # suggested tile width if saved as tiled

class ImageInfo(NamedTuple):
    width: int
    height: int
    components: int
    bits_per_pixel: int

class Loop(TypedDict):
    type: Union[
        Literal["TimeLoop"],
        Literal["XYPosLoop"],
        Literal["ZStackLoop"],
        Literal["NETimeLoop"],
    ]
    count: int
    nestingLevel: int
    parameters: dict

class ND2File:
    path: str
    def __init__(self, path: str) -> None: ...
    def is_open(self) -> bool: ...
    def open(self, path: str) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> ND2File: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool: ...
    def attributes(self) -> Attributes: ...
    @overload
    def metadata(self, *, format: Literal[True] = ...) -> structures.Metadata: ...
    @overload
    def metadata(self, *, format: Literal[False]) -> dict: ...
    @overload
    def metadata(
        self, frame: int, *, format: Literal[True] = ...
    ) -> structures.FrameMetadata: ...
    @overload
    def metadata(self, frame: int, *, format: Literal[False]) -> dict: ...
    def frame_metadata(self, seq_index: int) -> dict: ...
    def text_info(self) -> dict: ...
    def experiment(self) -> List[Loop]: ...
    def seq_count(self) -> int: ...
    def coord_size(self) -> int: ...
    def seq_index_from_coords(self, coords: Sequence[int]) -> int: ...
    def coords_from_seq_index(self, seq_index: int) -> Tuple[int, ...]: ...
    def image_info(self, seq_index: int = 0) -> ImageInfo: ...
    def coord_info(self) -> List[Tuple[int, str, int]]: ...
    def data(self, seq_index: int = 0) -> np.ndarray: ...

def imread(file: str = None, sequence: int = 0) -> np.ndarray: ...
