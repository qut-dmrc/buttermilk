#!/bin/bash
echo starting flow daemon
cd ${APP_PATH}
uv run python -m buttermilk.runner.cli --config-dir=/conf ${FLOW_PARAMS}
