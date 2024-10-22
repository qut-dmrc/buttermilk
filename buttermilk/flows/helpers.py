from pathlib import Path
from buttermilk.utils.utils import list_files_with_content


BASE_DIR = Path(__file__).absolute()
TEMPLATE_PATHS = [BASE_DIR.parent / "common", BASE_DIR.parent / "templates"]

def get_templates(pattern: str = ".*"):
    templates = list_files_with_content(TEMPLATE_PATHS, pattern=pattern)
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates