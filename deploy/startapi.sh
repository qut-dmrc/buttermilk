#!/bin/bash
echo starting flow daemon
uv run python -m buttermilk.runner.cli --config-dir=/conf ${FLOW_VARS}
