from typing import Any

from autogen_core.tools import Tool

from buttermilk._core.config import ToolConfig


def create_tool_functions(tool_cfg: dict[str, Any]) -> list[Tool]:
    """Instantiate tools and return their wrapped entry functions.
    
    Args:
        tool_cfg: Dictionary of tool configurations, keyed by tool name.
                 Values can be ToolConfig objects or direct tool instances.
    
    Returns:
        A list of Tool objects for consistency with autogen's tool system.
    """
    tools = []
    from buttermilk._core.log import logger

    for name, cfg in tool_cfg.items():
        try:
            # Check if it's already a tool instance (has get_tool or similar method)
            if hasattr(cfg, "get_tool"):
                # Direct tool instance - call get_tool() to get the FunctionTool
                tool = cfg.get_tool()
                if isinstance(tool, Tool):
                    tools.append(tool)
                else:
                    logger.warning(f"Tool {name} get_tool() returned non-Tool object: {type(tool)}")
            elif hasattr(cfg, "config") and hasattr(cfg, "__call__"):
                # It's a callable with config - might be a ToolConfig interface
                fn_list = cfg.config if isinstance(cfg.config, list) else [cfg.config]
                for fn in fn_list:
                    if isinstance(fn, Tool):
                        tools.append(fn)
            elif isinstance(cfg, Tool):
                # Direct Tool instance (e.g., FunctionTool)
                tools.append(cfg)
            elif isinstance(cfg, ToolConfig):
                # Traditional ToolConfig object
                from buttermilk.tools import AVAILABLE_TOOLS
                obj = AVAILABLE_TOOLS[str(cfg.tool_obj).lower()]
                tool = obj(**cfg.model_dump())
                fn_list = tool.get_functions()

                # Ensure all returned items are Tool instances
                for fn in fn_list:
                    if isinstance(fn, Tool):
                        tools.append(fn)
                    else:
                        logger.warning(f"Tool {name} ({cfg.tool_obj}) returned non-Tool object: {type(fn)}")
            else:
                logger.warning(f"Tool {name} has unknown configuration type: {type(cfg)}")

        except Exception as e:
            raise ValueError(f"Unable to instantiate tool '{name}': {cfg}: {e}") from e
    return tools
