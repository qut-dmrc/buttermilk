

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from promptflow.core import tool

BASE_DIR = Path(__file__).absolute().parent

TEMPLATE_PATHS = [BASE_DIR, BASE_DIR.parent / "common"]
@tool
def make_prompt(template: str) -> str:
    env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True)

    return env.get_template(template).render()

