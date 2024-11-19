import asyncio
from collections.abc import Mapping

import hydra
from pydantic import BaseModel

from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import BM
from buttermilk.runner.creek import Creek


class _CFG(BaseModel):
    bm: BM
    save: Mapping
    flows: dict[str, Creek]
    flow: str
    record: FlowRequest


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: _CFG) -> None:
    from rich import print as rprint

    # Hydra will automatically instantiate the objects
    # This should work, but doesn't?
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm

    record = FlowRequest(**cfg.record)
    flow = cfg.flow

    async def test_run_flow(creek: Creek):
        async for response in flow_stream(
            flow=creek,
            flow_request=record,
        ):
            rprint(response)

    asyncio.run(test_run_flow(objs.flows[flow]))


if __name__ == "__main__":
    main()
