import asyncio
from collections.abc import Mapping

import hydra
from pydantic import BaseModel
from rich import print as rprint

from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import BM
from buttermilk.runner.creek import Creek


class _CFG(BaseModel):
    bm: BM
    save: Mapping
    flows: dict[str, Creek]
    flow: str
    q: str
    record: dict


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: _CFG) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    params = {k: v for k, v in cfg.items() if k in {"record", "q"} and v}
    record = FlowRequest(**params, source="cli")
    flow = cfg.flow

    async def run_flow(creek: Creek):
        async for response in flow_stream(
            flow=creek,
            flow_request=record,
        ):
            rprint(response)

    asyncio.run(run_flow(objs.flows[flow]))


if __name__ == "__main__":
    main()
