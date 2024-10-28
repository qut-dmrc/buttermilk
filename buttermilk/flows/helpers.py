from pathlib import Path
from buttermilk.utils.utils import list_files_with_content
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined


BASE_DIR = Path(__file__).absolute()
TEMPLATE_PATHS = [BASE_DIR.parent / "templates"]

def get_templates(pattern: str = '', parent: str='', extension: str=''):
    templates = list_files_with_content(TEMPLATE_PATHS, pattern=pattern, parent=parent, extension=extension)
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates
class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'