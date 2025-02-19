import asyncio
from collections.abc import Mapping

import hydra
from pydantic import BaseModel
from rich import print as rprint

from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import BM
from buttermilk.runner.flow import Flow


class _CFG(BaseModel):
    bm: BM
    save: Mapping
    flows: dict[str, Flow]
    flow: str
    q: str
    record: dict


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    params = {k: v for k, v in cfg.items() if k in {"record", "record_id", "q"} and v}
    record = FlowRequest(**params, source="cli")
    flow = cfg.flow

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1

    async def run_flow(flow: Flow):
        async for response in flow_stream(
            flow=flow,
            flow_request=record,
        ):
            rprint(response)

    asyncio.run(run_flow(objs.flows[flow]))


if __name__ == "__main__":
    main()
