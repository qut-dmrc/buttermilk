
from typing import Generator, Any
from promptflow import tool

@tool
def select_model(models: list[str]) -> Generator[str, list[str], None]:
    for model in models:
        yield model
