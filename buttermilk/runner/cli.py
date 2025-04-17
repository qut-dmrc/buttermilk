import asyncio
import os
import threading

import hydra
from omegaconf import OmegaConf
from rich import print
from buttermilk._core import StepRequest
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk.bm import BM
from buttermilk.runner.chat import Selector
from buttermilk.runner.groupchat import AutogenOrchestrator
from buttermilk.runner.slackbot import register_handlers
import uvicorn

ORCHESTRATOR_CLASSES = {"simple": AutogenOrchestrator, "selector": Selector}

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: OrchestratorProtocol) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm: BM = objs.bm

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1
    match objs.ui:
        case "console":
            flow_name = cfg.flow
            orchestrator = ORCHESTRATOR_CLASSES[cfg.orchestrator](bm=bm, **objs.flows[flow_name])

            if record_id_or_url := cfg.record:
                request = StepRequest(role="fetch", prompt=record_id_or_url, description="")
                asyncio.run(orchestrator.run(request=request))
            else:
                asyncio.run(orchestrator.run())

        case "api":
            from buttermilk.api.flow import app

            # Store the necessary state in the app instance
            app.state.flows = objs.flows
            app.state.bm = bm
            app.state.orchestrators = ORCHESTRATOR_CLASSES

            bm.logger.info("API starting...")
            uvicorn.run(app, host="0.0.0.0", port=8000)

        case "pub/sub":
            from buttermilk.api.pubsub import start_pubsub_listener

            listener_thread = threading.Thread(target=start_pubsub_listener)
            listener_thread.start()

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
            slack_app, handler = initialize_slack_bot(
                bot_token=bot_token,
                app_token=app_token,
                loop=loop,
            )
            t = loop.create_task(handler.start_async())

            async def runloop():
                await register_handlers(
                    slack_app=slack_app,
                    flows=objs.flows,
                    orchestrator_tasks=orchestrator_tasks,
                )
                while True:
                    await asyncio.sleep(1)

            loop.run_until_complete(runloop())

            bm.logger.info("Slackbot exiting...")


if __name__ == "__main__":
    main()
