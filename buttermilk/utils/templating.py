from typing import Tuple

from buttermilk._core.runner_types import RecordInfo
from buttermilk.defaults import TEMPLATE_PATHS
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.utils import list_files_with_content, list_files
from jinja2 import FileSystemLoader, Undefined
from buttermilk import logger
import regex as re

from jinja2 import TemplateNotFound
from buttermilk.utils.utils import read_text
from jinja2.sandbox import SandboxedEnvironment


def get_templates(pattern: str = "", parent: str = "", extension: str = ""):
    templates = list_files_with_content(
        TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension
    )
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates


def get_template_names(pattern: str = "", parent: str = "", extension: str = "jinja2"):
    return [
        file_path.stem
        for file_path in list_files(
            TEMPLATE_PATHS, filename=pattern, parent=parent, extension=extension
        )
    ]


class KeepUndefined(Undefined):
    def __str__(self):
        return "{{ " + self._undefined_name + " }}"


def _parse_prompty(string_template) -> str:
    # Use Promptflow's format and strip the header out
    pattern = r"-{3,}\n(.*)-{3,}\n(.*)"
    result = re.search(pattern, string_template, re.DOTALL)
    if not result:
        return string_template
    else:
        return result.group(2)


def load_template_vars(
    *,
    template: str,
    **inputs,
) -> str:
    recursive_paths = TEMPLATE_PATHS + [
        x for p in TEMPLATE_PATHS for x in p.rglob("*") if x.is_dir()
    ]
    loader = FileSystemLoader(searchpath=recursive_paths)

    env = SandboxedEnvironment(
        loader=loader,
        trim_blocks=True,
        keep_trailing_newline=True,
        undefined=KeepUndefined,
    )

    # Read main template and strip out Prompty header
    filename = env.get_template(f"{template}.jinja2").filename
    logger.debug(f"Loading template {template} from {filename}.")
    tpl_text = read_text(filename)
    tpl_text = _parse_prompty(tpl_text)

    # Load template variables into the Jinja2 environment
    for k, v in inputs.items():
        if isinstance(v, str) and v:
            # Try to load a template if it's passed in by filename, otherwise use it
            # as a plain string replacement.

            try:
                filename = env.get_template(f"{v}.jinja2").filename
                var = read_text(filename)
                var = _parse_prompty(var)
                env.globals[k] = env.from_string(var).render()
                logger.debug(f"Loaded template variable {k} from {filename}.")
            except TemplateNotFound:
                # Leave the value as is and pass it in as text
                # Note here we don't treat this as a Jinja2 template to interpret;
                # the only templates we trust are the ones we read from disk.
                pass

    # Compile and render the templates, leaving unfilled variables to substitute later
    rendered_template = env.from_string(tpl_text).render()

    # From this point, langchain expects single braces for replacement instead of double
    # we could do this with a custom template class, but it's easier to just do it here.
    rendered_template = re.sub(r"{{\s+([a-zA-Z0-9_]+)\s+}}", r"{\1}", rendered_template)

    # We now have a template formatted as a string in Prompty format
    return rendered_template

def fill_placeholders(model_capabilities: LLMCapabilities, **input_vars):
    # Fill placeholders
    for k, v in input_vars.items():
        if isinstance(v, RecordInfo):
            if rendered := v.as_langchain_message(role="user", model_capabilities=model_capabilities):
                input_vars[k] = [rendered]
        elif v and v[0]:
            input_vars[k] = v

def make_messages(local_template: str) -> list[Tuple[str, str]]:
    try:
        # Parse messages using Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(local_template)

        # Next we use Prompty's format to set roles within the template
        from promptflow.core._prompty_utils import parse_chat

        messages = parse_chat(
            prompty, valid_roles=["system", "user", "human", "placeholder"]
        )

    except Exception as e:
        msg = f"Unable to decode template expecting Prompty format: {e}, {e.args=}"
        raise (ValueError(msg)) from e

    return messages
