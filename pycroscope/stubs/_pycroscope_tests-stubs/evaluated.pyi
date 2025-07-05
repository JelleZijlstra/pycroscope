from typing import IO, Any, BinaryIO, TextIO

from pycroscope.extensions import evaluated

@evaluated
def open(mode: str):
    if mode == "r":
        return TextIO
    elif mode == "rb":
        return BinaryIO
    else:
        return IO[Any]

@evaluated
def open2(mode: str) -> IO[Any]:
    if mode == "r":
        return TextIO
    elif mode == "rb":
        return BinaryIO
