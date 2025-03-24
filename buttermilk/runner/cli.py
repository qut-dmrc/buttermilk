import asyncio
import os

import hydra

from buttermilk._core.contract import OrchestratorProtocol
from buttermilk.bm import BM
from buttermilk.runner.autogen import AutogenOrchestrator
from buttermilk.runner.chat import Selector
from buttermilk.runner.simple import Sequencer

orchestrators = [Sequencer, AutogenOrchestrator, Selector]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: OrchestratorProtocol) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm: BM = objs.bm

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1

    match objs.entry:
        case "console":
            flow_name = cfg.flow
            orchestrator_name = objs.flows[flow_name].pop("orchestrator")
            orchestrator = globals()[orchestrator_name](**objs.flows[flow_name])
            asyncio.run(orchestrator.run())
        case "slackbot":
            bm.logger.info("Slackbot starting...")
            bot_token = bm.credentials["MODBOT_TOKEN"]
            os.environ["SLACK_BOT_TOKEN"] = bot_token
            app_token = bm.credentials["SLACK_APP_TOKEN"]
            os.environ["SLACK_APP_TOKEN"] = app_token

            from buttermilk.runner.slackbot import initialize_slack_bot

            loop = asyncio.get_event_loop()
            loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1

            orchestrator_tasks: asyncio.Queue[asyncio.Task] = asyncio.Queue()
            # Start the slack bot, which has its own triggers to respond to events
            handler = initialize_slack_bot(
                bot_token=bot_token,
                app_token=app_token,
                flows=objs.flows,
                loop=loop,
                orchestrator_tasks=orchestrator_tasks,
            )
            t = loop.create_task(handler.start_async())

            async def runloop():
                while True:
                    await asyncio.sleep(1)

            loop.run_until_complete(runloop())

            bm.logger.info("Slackbot exiting...")


if __name__ == "__main__":
    main()
