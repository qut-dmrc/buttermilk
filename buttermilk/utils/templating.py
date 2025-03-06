from collections.abc import Sequence

import regex as re
from jinja2 import FileSystemLoader, TemplateNotFound, Undefined
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.messages import HumanMessage

from buttermilk import logger
from buttermilk._core.runner_types import RecordInfo
from buttermilk.defaults import TEMPLATE_PATHS
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.utils import list_files, list_files_with_content, read_text


def get_templates(pattern: str = "", parent: str = "", extension: str = ""):
    templates = list_files_with_content(
        TEMPLATE_PATHS,
        filename=pattern,
        parent=parent,
        extension=extension,
    )
    templates = [(t.replace(".jinja2", ""), tpl) for t, tpl in templates]
    return templates


def get_template_names(pattern: str = "", parent: str = "", extension: str = "jinja2"):
    return [
        file_path.stem
        for file_path in list_files(
            TEMPLATE_PATHS,
            filename=pattern,
            parent=parent,
            extension=extension,
        )
    ]


def _parse_prompty(string_template) -> str:
    # Use Promptflow's format and strip the header out
    pattern = r"-{3,}\n(.*)-{3,}\n(.*)"
    result = re.search(pattern, string_template, re.DOTALL)
    if not result:
        return string_template
    return result.group(2)


def load_template_vars(
    *,
    template: str,
    parameters: dict,
    **inputs,
) -> tuple[str, list]:
    recursive_paths = TEMPLATE_PATHS + [
        x for p in TEMPLATE_PATHS for x in p.rglob("*") if x.is_dir()
    ]
    loader = FileSystemLoader(searchpath=recursive_paths)

    undefined_vars = []

    class KeepUndefined(Undefined):
        def __str__(self):
            # Keep a list of variables that have not yet been filled.
            undefined_vars.append(self._undefined_name)

            # We leave double races here, so remember to use
            # format as jinja2 methods so that json instructions and examples
            # etc are not misinterpreted as variables.
            return "{{" + self._undefined_name + "}}"

    env = SandboxedEnvironment(
        loader=loader,
        trim_blocks=True,
        keep_trailing_newline=True,
        # undefined=KeepUndefined,
    )

    # Read main template and strip out Prompty header
    filename = env.get_template(f"{template}.jinja2").filename
    logger.debug(f"Loading template {template} from {filename}.")
    tpl_text = read_text(filename)
    tpl_text = _parse_prompty(tpl_text)
    available_vars = {}
    # Load template variables into the Jinja2 environment
    for k, v in parameters.items():
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
                available_vars[k] = v
        else:
            available_vars[k] = v

    available_vars.update(inputs)

    # Compile and render the templates, leaving unfilled variables to substitute later
    tpl = env.from_string(tpl_text)
    rendered_template = tpl.render(**available_vars)
    undefined_vars = [x for x in undefined_vars if x not in available_vars]

    # We now have a template formatted as a string in Prompty format.
    # Also return the leftover (unfilled) inputs to pass through later
    return rendered_template, undefined_vars


def prepare_placeholders(model_capabilities: LLMCapabilities, **input_vars) -> dict:
    # Fill placeholders
    placeholders = {}
    for k, v in input_vars.items():
        if isinstance(v, RecordInfo):
            if rendered := v.as_langchain_message(
                role="user",
                model_capabilities=model_capabilities,
            ):
                placeholders[k] = [rendered]
        elif isinstance(v, str):
            placeholders[k] = [HumanMessage(v)]
        elif isinstance(v, Sequence):
            # Lists may need to be handled separately...?
            placeholders[k] = "\n\n".join(v)
            placeholders[k] = [HumanMessage(v)]
        elif v:
            placeholders[k] = [HumanMessage(v)]

    return placeholders


def make_messages(local_template: str) -> list[tuple[str, str]]:
    lc_messages = []
    try:
        # Parse messages using Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(local_template)

        # Next we use Prompty's format to set roles within the template
        from promptflow.core._prompty_utils import parse_chat

        messages = parse_chat(
            prompty,
            valid_roles=["system", "user", "human", "placeholder"],
        )

        # Convert to langchain format
        # (Later we won't need this, because langchain ends up converting back to our json anyway)
        for message in messages:
            role = message["role"]
            content = message["content"]
            if content:
                # Don't add empty messages
                lc_messages.append((role, content))

    except Exception as e:
        msg = f"Unable to decode template expecting Prompty format: {e}, {e.args=}"
        raise (ValueError(msg)) from e

    return lc_messages
