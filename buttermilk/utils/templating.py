from typing import Any

import regex as re
from jinja2 import (
    FileSystemLoader,
    TemplateNotFound,
    Undefined,
    sandbox,
)
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk.defaults import TEMPLATE_PATHS
from buttermilk.exceptions import FatalError
from buttermilk.utils.utils import list_files, list_files_with_content, read_text


class KeyValueCollector(BaseModel):
    """A simple collector for key-value pairs
    to insert into templates.
    """

    _data: dict[str, Any | list[Any]] = PrivateAttr(default_factory=dict)

    def update(self, incoming: dict) -> None:
        for key, value in incoming.items():
            self.add(key, value)

    def add(
        self,
        key: str,
        value: Any,
    ) -> None:
        if key in self._data:
            if not isinstance(self._data[key], list):
                self._data[key] = [self._data[key]]
            self._data[key].append(value)
        else:
            self._data[key] = [value]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get_dict(self) -> dict:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

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
    parameters: dict,
    untrusted_inputs: dict = {},
) -> tuple[str, set[str]]:
    """Render a template with hierarchical includes and some limited security controls.

    Args:
        parameters:         Parameters you control (allows file includes)
        untrusted_inputs:   User-provided inputs (restricted to string interpolation)

    Returns:
        Tuple: Fully rendered template; set of unfilled variables.

    """
    template = parameters["template"]

    recursive_paths = TEMPLATE_PATHS + [
        x for p in TEMPLATE_PATHS for x in p.rglob("*") if x.is_dir()
    ]
    loader = FileSystemLoader(searchpath=recursive_paths)

    undefined_vars = []

    class KeepUndefined(Undefined):
        def __str__(self):
            # Keep a list of variables that have not yet been filled.
            undefined_vars.append(self._undefined_name)

            # We leave double braces here, that json instructions and
            # examples etc are not misinterpreted as variables.
            return "{{" + str(self._undefined_name) + "}}"

    # Create a sandbox environment for template processing
    sandbox_env = sandbox.SandboxedEnvironment(
        loader=loader,
        trim_blocks=True,
        undefined=KeepUndefined,  # Retain unfilled placeholders
        keep_trailing_newline=False,
    )

    # Load main template
    try:
        filename = str(sandbox_env.get_template(f"{template}.jinja2").filename)
    except Exception as err:
        raise FatalError(f"Template {template} not found.") from err

    logger.debug(f"Loading template {template} from {filename}.")
    tpl_text = read_text(filename)
    tpl_text = _parse_prompty(tpl_text)

    # Process template inclusions in parameters
    processed_params = {}
    for k, v in parameters.items():
        if isinstance(v, str) and v:
            try:
                # Try to load this as a template
                filename = sandbox_env.get_template(f"{v}.jinja2").filename
                var = read_text(filename)
                var = _parse_prompty(var)
                processed_params[k] = var
                logger.debug(f"Loaded template variable {k} from {filename}.")
            except TemplateNotFound:
                # Not a template, use as regular string
                processed_params[k] = v
        else:
            processed_params[k] = v

    # Create a combined variables dict with trusted parameters taking precedence
    all_vars = dict(untrusted_inputs)
    all_vars.update(processed_params)  # Trusted params override untrusted ones

    # Render everything in one pass
    tpl = sandbox_env.from_string(tpl_text)
    rendered = tpl.render(**all_vars)

    return rendered, set(undefined_vars)


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
