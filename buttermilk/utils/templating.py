from pathlib import Path
from typing import Any, Mapping, Sequence

import regex as re
from jinja2 import (
    FileSystemLoader,
    TemplateNotFound,
    Undefined,
    sandbox,
)
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from buttermilk._core.defaults import TEMPLATES_PATH
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.types import Record
from buttermilk.utils.utils import list_files, list_files_with_content, read_text

from buttermilk._core.contract import AllMessages, AssistantMessage, LLMMessage, SystemMessage, UserMessage

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
        if not (isinstance(value, list) and not isinstance(value, str)):
            value = [value]

        if key in self._data:
            self._data[key].extend(value)
        else:
            self._data[key] = value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get_dict(self) -> dict:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def init(self, keys: list[str]) -> None:
        for key in keys:
            self._data[key] = []
    """Routes variables between workflow steps using mappings

    Data is essentially a dict of lists, where each key is the name of a step and
    each list is the output of an agent in a step.
    """

    _data: dict[str, Any] = PrivateAttr(default_factory=dict)

    def _resolve_mappings(self, mappings: dict[Any, Any]) -> dict[str, Any]:
        """Resolve all variable mappings to their values"""
        resolved = {}

        if isinstance(mappings, str):
            # We have reached the end of the recursive road
            return self._resolve_simple_path(mappings)

        for target, source_spec in mappings.items():
            if isinstance(source_spec, Sequence) and not isinstance(source_spec, str):
                # Handle aggregation case
                results = []
                for src in source_spec:
                    result = self._resolve_mappings(src)
                    if result:
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                resolved[target] = results
            elif isinstance(source_spec, Mapping | dict):
                # Handle nested mappings
                resolved[target] = self._resolve_mappings(source_spec)
            else:
                resolved[target] = self._resolve_simple_path(source_spec)

        # remove empty values, but keep in empty lists etc where we found the
        # variable to substitute but it was empty.
        resolved = {k: v for k, v in resolved.items() if v is not None}

        return resolved

    def _resolve_simple_path(self, path: str) -> Any:
        """Resolve a simple dot-notation path

        When a step has multiple outputs, returns a list with all matching results.
        For JMESPath expressions that return lists, flattens the results.
        """
        if not path:
            return None
        if "." not in path:
            # Direct reference to a step's complete output list
            return self._data.get(path, None)

        # Handle dot notation for nested fields
        step_name, field_path = path.split(".", 1)

        if step_name not in self._data:
            return None

        # Get all outputs for this step
        step_results = self._data[step_name]
        if not step_results:
            return None

        # Collect all matching results from all outputs
        all_matches = []

        for result in step_results:
            # Convert Pydantic models to dictionaries for JMESPath
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = result

            value = jmespath.search(field_path, result_dict)
            if value is not None:
                # If the value is already a list, extend our results
                if isinstance(value, list):
                    all_matches.extend(value)
                else:
                    all_matches.append(value)

        # If no match was found, return None
        if not all_matches:
            return None

        return all_matches


def get_templates(pattern: str = "", parent: str = "", extension: str = ""):
    templates = list_files_with_content(
        TEMPLATES_PATH,
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
            TEMPLATES_PATH,
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


def load_template(template: str,
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

    recursive_paths = [TEMPLATES_PATH] + [
        p for p in Path(TEMPLATES_PATH).rglob("*") if p.is_dir()
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

    # Create a combined variables dict with trusted parameters taking precedence
    all_vars = dict(untrusted_inputs)
    all_vars.update(processed_params)  # Trusted params override untrusted ones

    # Render everything in one pass
    tpl = sandbox_env.from_string(tpl_text)
    rendered = tpl.render(**all_vars)

    return rendered, set(undefined_vars)


def make_messages(
    local_template: str, context: list[LLMMessage], records: list[Record], fail_on_missing_placeholders: bool = False
) -> list[LLMMessage]:
    output: list[LLMMessage] = []
    try:
        # Parse messages using Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(local_template)

    except Exception as e:
        msg = f"Unable to decode template expecting Prompty format: {e}, {e.args=}"
        raise (ValueError(msg)) from e

    # Next we use Prompty's format to set roles within the template
    from promptflow.core._prompty_utils import parse_chat

    messages = parse_chat(
        prompty,
        valid_roles=["system", "user", "assistant", "placeholder",
                "developer",
                "human"]
    )

    # Convert to LLMMessage
    for message in messages:
        var_name = re.sub(r"[^\w\d_]+", "", message["content"]).lower()
        if not var_name:
            # don't add empty messages
            continue 
        match message['role'].lower():
            case "developer" | "system":
                output.append(SystemMessage(content=message["content"]))
            case "user" | "human":            
                output.append(UserMessage(content=message["content"], source="template"))                      
            case "assistant":
                output.append(AssistantMessage(content=message["content"], source="template"))
            case "placeholder":
                # Remove everything except word chars to get the variable name
                if var_name == "context":
                    # special case for context
                    output.extend(context)
                elif var_name == "records":
                    output.extend([rec.as_message() for rec in records])
                else:
                    err = (f"Missing {var_name} in placeholder vars.",)
                    raise ProcessingError(err)
            case _:
                err = (
                        f"Missing {var_name} in placeholder vars.",
                    )
                if fail_on_missing_placeholders:
                    raise ProcessingError(err)
                logger.warning(err)

    return output
