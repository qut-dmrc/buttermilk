
from typing import Any, List, Optional, TypedDict

from promptflow import tool
from promptflow.core import ToolProvider, tool

from buttermilk.lc import LC


class LangChainLooper(ToolProvider):
    def __init__(self, *, models: dict, template_path: str, other_templates: dict = {}, other_vars: Optional[dict] = None) -> None:

        self.lc = LC(models=models, template=template_path, other_templates=other_templates, other_vars=other_vars)

    @tool
    def __call__(self):
        pass