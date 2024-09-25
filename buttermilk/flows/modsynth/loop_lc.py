
from promptflow import tool
from typing import Any, List, Optional, TypedDict
from buttermilk.flows.lc.lc import LangChainMulti
from promptflow.core import (
    ToolProvider,
    tool
)

class LangChainLooper(ToolProvider):
    def __init__(self, *, models: dict, template_path: str, other_templates: dict = {}, other_vars: Optional[dict] = None) -> None:

        self.lc = LangChainMulti(models=models, template_path=template_path, other_templates=other_templates, other_vars=other_vars)

    @tool
    def __call__(