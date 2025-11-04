#!/usr/bin/env python
"""Debug script to check logger configuration."""

import logging

from buttermilk import init

bm = init(config_name="testing")

print("\n=== Logger Hierarchy ===")
print(f"Root logger handlers: {logging.getLogger().handlers}")
print(f'Buttermilk logger handlers: {logging.getLogger("buttermilk").handlers}')
print(f'Buttermilk logger level: {logging.getLevelName(logging.getLogger("buttermilk").level)}')
print(f'Buttermilk logger propagate: {logging.getLogger("buttermilk").propagate}')

print("\n=== Handler Details ===")
for i, handler in enumerate(logging.getLogger("buttermilk").handlers):
    print(f"Handler {i}: {type(handler).__name__}")
    print(f"  Level: {logging.getLevelName(handler.level)}")
    if hasattr(handler, "name"):
        print(f"  Name: {handler.name}")

# Test logging a message
from buttermilk import logger

logger.info("Test message from debug script", test_marker="debug")

print("\n=== Logger Config ===")
print(f"Logger cfg: {bm._logger_cfg}")
print(f"Logger cfg type: {bm._logger_cfg.type if bm._logger_cfg else None}")
