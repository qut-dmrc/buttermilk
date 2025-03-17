
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.ui import IOInterface

BASE_DIR = Path(__file__).absolute().parent

class FlowProtocol(Protocol):
    save: SaveInfo
    data: Sequence[DataSource]
    steps: Sequence["Agent"]
    interface: IOInterface

    async def run(self, job: "Job") -> "Job":
        ...

    async def __call__(self, job: "Job") -> "Job":
        ...


class OrchestratorProtocol(Protocol):
    bm: "BM"
    flows: Mapping[str, FlowProtocol]
