import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Self

from autogen_core import FunctionCall
import regex as re

from buttermilk._core.agent import Agent, CancellationToken, ToolConfig
from buttermilk._core.runner_types import Record
from buttermilk.agents.llm import FunctionTool, Tool, ToolSchema

def create_tool_functions(tool_cfg: list[ToolConfig]) -> list[FunctionCall | Tool | ToolSchema | FunctionTool]:
    """Instantiate tools and return their wrapped entry functions."""
    _fns = []
    for cfg in tool_cfg:
        try:
            fn = create_tool_function(cfg)
            _fns.append(fn)
        except Exception as e:
            raise ValueError(f"Unable to instantiate tool: {cfg}: {e}") from e
    return _fns

def create_tool_function(cfg: ToolConfig) -> FunctionCall | Tool | ToolSchema | FunctionTool:
    """Create function for a single tool."""
    
    from buttermilk.tools import AVAILABLE_TOOLS

    obj = AVAILABLE_TOOLS[str(cfg.tool_obj).lower()]
    tool = obj(**cfg.model_dump())
    fn = FunctionTool(
        tool._run,
        description=cfg.description, 
        name=cfg.name,strict=False,
    )
    return fn
