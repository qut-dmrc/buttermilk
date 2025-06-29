from autogen_core.tools import Tool

from buttermilk._core.config import ToolConfig


def create_tool_functions(tool_cfg: list[ToolConfig]) -> list[Tool]:
    """Instantiate tools and return their wrapped entry functions.
    
    Returns a list of Tool objects for consistency with autogen's tool system.
    """
    tools = []
    for cfg in tool_cfg:
        try:
            from buttermilk.tools import AVAILABLE_TOOLS

            obj = AVAILABLE_TOOLS[str(cfg.tool_obj).lower()]
            tool = obj(**cfg.model_dump())
            fn_list = tool.get_functions()
            
            # Ensure all returned items are Tool instances
            for fn in fn_list:
                if isinstance(fn, Tool):
                    tools.append(fn)
                else:
                    # Log a warning if we get non-Tool objects
                    from buttermilk._core.log import logger
                    logger.warning(f"Tool {cfg.tool_obj} returned non-Tool object: {type(fn)}")
                    
        except Exception as e:
            raise ValueError(f"Unable to instantiate tool: {cfg}: {e}") from e
    return tools
