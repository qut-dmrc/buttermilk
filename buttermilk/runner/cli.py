import asyncio

import hydra
from rich import print as rprint

from buttermilk._core.contract import OrchestratorProtocol
from buttermilk.api.stream import flow_stream
from buttermilk.runner.autogen import AutogenOrchestrator
from buttermilk.runner.flow import Flow
from buttermilk.runner.simple import Sequencer

orchestrators = [Sequencer, AutogenOrchestrator]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: OrchestratorProtocol) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    flow_name = cfg.flow

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1

    orchestrator_name = objs.flows[flow_name].pop("orchestrator")
    orchestrator = globals()[orchestrator_name](**objs.flows[flow_name])
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
