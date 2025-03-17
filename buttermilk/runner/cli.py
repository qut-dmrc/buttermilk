import asyncio
from collections.abc import Mapping

import hydra
from pydantic import BaseModel
from rich import print as rprint

from buttermilk._core.contract import OrchestratorProtocol
from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import bm
from buttermilk.runner.flow import Flow



@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: OrchestratorProtocol) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    flow_name = cfg.flow

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1
    
    async def run_flow(flow: Flow):
        async for response in flow_stream(
            flow=flow,
            flow_request=record,
            return_json=False
        ):
            rprint(response)

    asyncio.run(objs.flows[flow_name].run())


if __name__ == "__main__":
    main()
