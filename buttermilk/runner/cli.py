import asyncio
import os
import threading

import hydra
from omegaconf import OmegaConf
from rich import print
from buttermilk._core.contract import StepRequest  # Keep for type hints if needed elsewhere
from buttermilk._core.types import RunRequest  # Import RunRequest
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk.agents.fetch import FetchRecord
from buttermilk.bm import BM  # Original import
from buttermilk.runner.selector import Selector
from buttermilk.runner.groupchat import AutogenOrchestrator
from buttermilk.runner.slackbot import register_handlers
import uvicorn

ORCHESTRATOR_CLASSES = {"simple": AutogenOrchestrator, "selector": Selector}


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: OrchestratorProtocol) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm: BM = objs.bm

    # Call the setup method now that bm has been populated by Hydra
    bm.setup_instance()  # Add the call to the setup method

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1
    match objs.ui:
        case "console":
            orchestrator: Orchestrator = objs.flows[cfg.flow]

            run_request = RunRequest(record_id=cfg.record_id or "", prompt=cfg.prompt or "", uri=cfg.uri or "")
            # Pass RunRequest (or None) to orchestrator
            asyncio.run(orchestrator.run(request=run_request))  # Original run

        case "api":
            from buttermilk.api.flow import app

            # Store the necessary state in the app instance
            app.state.flows = objs.flows
            app.state.orchestrators = ORCHESTRATOR_CLASSES

            bm.logger.info("API starting...")
            uvicorn.run(app, host="0.0.0.0", port=8000)  # Original uvicorn run

        case "pub/sub":
            from buttermilk.api.pubsub import start_pubsub_listener

            listener_thread = threading.Thread(target=start_pubsub_listener)  # Original thread start
            listener_thread.start()

        case "slackbot":
            bm.logger.info("Slackbot starting...")
            # Original credential access - might still have type errors if bm.credentials isn't dict
            # Add type check before accessing credentials
            creds = bm.credentials
            if not isinstance(creds, dict):
                raise TypeError(f"Expected credentials to be a dict, got {type(creds)}")

            bot_token = creds.get("MODBOT_TOKEN")
            app_token = creds.get("SLACK_APP_TOKEN")

            if not bot_token or not app_token:
                raise ValueError("Missing MODBOT_TOKEN or SLACK_APP_TOKEN in credentials.")

            os.environ["SLACK_BOT_TOKEN"] = bot_token
            os.environ["SLACK_APP_TOKEN"] = app_token

            from buttermilk.runner.slackbot import initialize_slack_bot

            loop = asyncio.get_event_loop()  # Original loop access
            loop.slow_callback_duration = 1.0  # Set to 1 second instead of default 0.1

            orchestrator_tasks: asyncio.Queue[asyncio.Task] = asyncio.Queue()
            # Start the slack bot, which has its own triggers to respond to events
            slack_app, handler = initialize_slack_bot(  # Original call
                bot_token=bot_token,
                app_token=app_token,
                loop=loop,
            )
            t = loop.create_task(handler.start_async())  # Original task creation

            async def runloop():  # Original async runloop
                await register_handlers(
                    slack_app=slack_app,
                    flows=objs.flows,
                    orchestrator_tasks=orchestrator_tasks,
                )
                while True:
                    await asyncio.sleep(1)

            loop.run_until_complete(runloop())  # Original loop run

            bm.logger.info("Slackbot exiting...")


if __name__ == "__main__":
    main()  # Original main call
