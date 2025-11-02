from contextlib import AbstractContextManager
from typing import AnyStr, TypeAlias

class _ScandirIterator(AbstractContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...

StrJson: TypeAlias = str | dict[str, StrJson]
