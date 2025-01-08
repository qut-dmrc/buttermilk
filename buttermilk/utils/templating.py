from pathlib import Path
from typing import Mapping, Optional, Tuple

import yaml
from buttermilk.defaults import TEMPLATE_PATHS
from buttermilk.utils.utils import list_files_with_content, list_files
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined
from buttermilk import logger
import regex as re


def get_templates(pattern: str = '', parent: str='', extension: str=''):
    templates = list_files_with_content(TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension)
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates

def get_template_names(pattern: str = '', parent: str='', extension: str='jinja2'):
    return  [ file_path.stem for file_path in list_files(TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension) ]

class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'
    
def _parse_prompty(string_template) -> str:
        # Use Promptflow's format and strip the header out
        pattern = r"-{3,}\n(.*)-{3,}\n(.*)"
        result = re.search(pattern, string_template, re.DOTALL)
        if not result:
            return string_template
        else:
            return result.group(2)

def make_messages(local_template: str) -> list[Tuple[str,str]]:

    try:
        # We'll aim to be compatible with Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(local_template)

        # Next we use Prompty's format to set roles within the template
        from promptflow.core._prompty_utils import parse_chat
        lc_messages = parse_chat(prompty, valid_roles=["system", "user", "human", "placeholder"])

        # Finally we export it out to Langchain's format.
        messages = []
        for message in lc_messages:
            role = message['role']
            content = message['content']
            messages.append((role, content))

    except Exception as e:
        # But will fall back to using full text if necessary
        logger.warning(f'Unable to decode template as Prompty: {e}, {e.args=}')
        messages = [('human', local_template)]

    return messages


