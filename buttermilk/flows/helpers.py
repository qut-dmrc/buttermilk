from pathlib import Path
from buttermilk.utils.utils import list_files_with_content, list_files
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined


BASE_DIR = Path(__file__).absolute()
TEMPLATE_PATHS = [BASE_DIR.parent / "templates"]

def get_templates(pattern: str = '', parent: str='', extension: str=''):
    templates = list_files_with_content(TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension)
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates

def get_template_names(pattern: str = '', parent: str='', extension: str='jinja2'):
    return  [ file_path.stem for file_path in list_files(TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension) ]

class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'