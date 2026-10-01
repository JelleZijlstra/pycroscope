from typing import NamedTuple

from typing_extensions import NamedTuple as ExtensionsNamedTuple

class Point(NamedTuple):
    x: int

class Child(Point): ...

class ExtensionPoint(ExtensionsNamedTuple):
    x: int

class CustomPoint(Point):
    def _asdict(self) -> dict[str, int]: ...
