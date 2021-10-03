from typing import TYPE_CHECKING, List, NewType, Tuple, Union, overload

from typing_extensions import Literal, Protocol
from wurlitzer import pipes

FileHandle = NewType("FileHandle", int)

if TYPE_CHECKING:
    import numpy as np

    # fmt: off
    class SDKModule(Protocol):
        @staticmethod
        def open(path: str) -> FileHandle: ...
        @staticmethod
        def close(fh: FileHandle) -> None: ...
        @staticmethod
        def get_seq_count(fh: FileHandle) -> int: ...
        @staticmethod
        def get_attributes(fh: FileHandle) -> dict: ...
        @staticmethod
        def get_experiment(fh: FileHandle) -> List[dict]: ...
        @staticmethod
        def get_metadata(fh: FileHandle) -> dict: ...
        @staticmethod
        def get_text_info(fh: FileHandle) -> dict: ...
        @staticmethod
        def get_image(fh: FileHandle, index: int) -> np.ndarray: ...
        @staticmethod
        def get_seq_index_from_coords(fh, coords: Union[List[int], Tuple[int, ...]]) -> int: ...
        @staticmethod
        def get_coords_from_seq_index(fh, seq_index: int) -> Tuple[int, ...]: ...
        @overload
        @staticmethod
        def get_coord_info(fh, seq_index: Literal[None] = None) -> List[Tuple[int, str, int]]: ...
        @overload
        @staticmethod
        def get_coord_info(fh, seq_index: int) -> Tuple[int, str, int]: ...
    # fmt: on

    latest: SDKModule
    v9: SDKModule

with pipes():
    from . import latest, v9


__all__ = ["latest", "v9"]
