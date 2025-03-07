from collections.abc import Sequence
from typing import Any

import regex as re
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    Undefined,
    sandbox,
)
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.runner_types import RecordInfo
from buttermilk.defaults import TEMPLATE_PATHS
from buttermilk.exceptions import FatalError
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.utils import list_files, list_files_with_content, read_text


class KeyValueCollector(BaseModel):
    """A simple collector for key-value pairs of messages
    to insert into templates.
    """

    _data: dict[str, Any | list[Any]] = PrivateAttr(default_factory=dict)

    def update(self, incoming: dict) -> None:
        for key, value in incoming.items():
            self.add(key, value)

    def add(
        self,
        key: str,
        value: UserMessage | SystemMessage | AssistantMessage,
    ) -> None:
        if key in self._data:
            if not isinstance(self._data[key], list):
                self._data[key] = [self._data[key]]
            self._data[key].append(value)
        else:
            self._data[key] = [value]

    def get_dict(self) -> dict:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


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


def load_template(
    *,
    template: str,
    parameters: dict = {},
) -> str:
    """First stage of a two-stage template rendering to safely handle
    trusted and untrusted variables.

    Args:
        template:           The template string
        parameters:         Parameters you control (allows file includes)
        untrusted_inputs:   User-provided inputs (restricted to string
                            interpolation only)

    Returns:    Rendered template string, with placeholders for
                unfilled variables.

    Raises:
        FatalError if the primary template is not found.

    """
    recursive_paths = TEMPLATE_PATHS + [
        x for p in TEMPLATE_PATHS for x in p.rglob("*") if x.is_dir()
    ]
    loader = FileSystemLoader(searchpath=recursive_paths)

    undefined_vars = []

    class KeepUndefined(Undefined):
        def __str__(self):
            # Keep a list of variables that have not yet been filled.
            undefined_vars.append(self._undefined_name)

            # We leave double braces here, so remember to use
            # format as jinja2 methods so that json instructions and examples
            # etc are not misinterpreted as variables.
            return "{{" + str(self._undefined_name) + "}}"

    trusted_env = Environment(
        autoescape=True,
        loader=loader,
        trim_blocks=True,
        keep_trailing_newline=True,
        # Leave some fields unfilled to be filled in later
        undefined=KeepUndefined,
    )

    # Read main template and strip out Prompty header
    try:
        filename = str(trusted_env.get_template(f"{template}.jinja2").filename)
    except Exception as err:
        raise FatalError(f"Template {template} not found.") from err

    logger.debug(f"Loading template {template} from {filename}.")
    tpl_text = read_text(filename)
    tpl_text = _parse_prompty(tpl_text)
    # Render with trusted params first

    available_vars = {}
    # Load template variables into the Jinja2 environment
    for k, v in parameters.items():
        if isinstance(v, str) and v:
            # Try to load a template if it's passed in by filename, otherwise use it
            # as a plain string replacement.

            try:
                filename = trusted_env.get_template(f"{v}.jinja2").filename
                var = read_text(filename)
                var = _parse_prompty(var)
                trusted_env.globals[k] = trusted_env.from_string(var).render()
                logger.debug(f"Loaded template variable {k} from {filename}.")
            except TemplateNotFound:
                # Leave the value as is and pass it in as text -- but in the next step.
                # The only templates we trust are the ones we read from disk.
                available_vars[k] = v
        else:
            available_vars[k] = v

    # Compile and render the templates, leaving unfilled variables to substitute later
    tpl = trusted_env.from_string(tpl_text)
    intermediate_template = tpl.render(**available_vars)

    return intermediate_template


def finalise_template(
    *,
    intermediate_template: str,
    untrusted_inputs: dict = {},
) -> str:
    """Second stage of a two-stage template rendering to safely handle
    trusted and untrusted variables.

    Args:
        intermediate_template:  A partial template
        untrusted_inputs:   User-provided inputs (restricted to string
                            interpolation only)

    Return:    Rendered template string, finalised.

    Raises:
        ValueError if any parameters are missing.

    """
    # Stage 2: Process user-provided inputs with separate sandbox
    sandbox_env = sandbox.SandboxedEnvironment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
    )

    # Render with untrusted inputs in sandbox
    final_rendered = sandbox_env.from_string(intermediate_template).render(
        **untrusted_inputs,
    )

    # We now have a template component formatted as a string.
    return final_rendered


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
