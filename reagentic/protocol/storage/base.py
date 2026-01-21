from __future__ import annotations

import abc
from typing import Dict, List

from ..models import ProtocolEntry


class ProtocolStorage(abc.ABC):
    @abc.abstractmethod
    async def write(self, entry: ProtocolEntry) -> None:
        pass

    @abc.abstractmethod
    async def write_batch(self, entries: List[ProtocolEntry]) -> None:
        pass

    @abc.abstractmethod
    async def read(self, session_id: str) -> List[ProtocolEntry]:
        pass

    @abc.abstractmethod
    async def query(self, filters: Dict) -> List[ProtocolEntry]:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass
