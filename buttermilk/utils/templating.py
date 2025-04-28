from pathlib import Path
from typing import Any, Mapping, Sequence

import jmespath
import regex as re
from jinja2 import (
    FileSystemLoader,
    Undefined,
    sandbox,
)
from pydantic import BaseModel, PrivateAttr

from buttermilk import logger
from autogen_core.models import AssistantMessage, UserMessage, LLMMessage, SystemMessage
from buttermilk._core.defaults import TEMPLATES_PATH
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.types import Record
from buttermilk.utils.utils import list_files, list_files_with_content


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
        if value is not None and value != [] and value != {} and value != "None":
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

    def _resolve_mappings(self, mappings: dict[Any, Any], data: Mapping) -> dict[str, Any]:
        """Resolve all variable mappings to their values"""

        ##
        ##
        ## TODO: I don't think this is needed anymore????
        ##
        ##
        resolved = {}

        # Convert Pydantic models to dictionaries for JMESPath
        if hasattr(data, "model_dump"):
            data_dict = data.model_dump()
        else:
            data_dict = data

        if isinstance(mappings, str):
            # We have reached the end of the recursive road
            return self._resolve_simple_path(mappings, data_dict)

        for target, source_spec in mappings.items():
            if isinstance(source_spec, Sequence) and not isinstance(source_spec, str):
                # Handle aggregation case
                results = []
                for src in source_spec:
                    result = self._resolve_mappings(src, data_dict)
                    if result:
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                resolved[target] = results
            elif isinstance(source_spec, Mapping | dict):
                # Handle nested mappings
                resolved[target] = self._resolve_mappings(source_spec, data_dict)
            else:
                resolved[target] = self._resolve_simple_path(source_spec, data_dict)

        # remove empty values and empty containers
        resolved = {k: v for k, v in resolved.items() if v is not None and v != {} and v != []}

        return resolved

    def _resolve_simple_path(self, path: str, data: Mapping) -> Any:
        """Resolve a JMESPath expression against the collected data."""
        if not path:
            return None

        try:
            # Directly search the entire data structure using the JMESPath expression
            result = jmespath.search(path, data)
            return result
        except Exception:
            # Optional: Log the error if the JMESPath expression is invalid or fails
            # logger.warning(f"JMESPath search failed for path '{path}': {e}")
            return None


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
    pattern = r"-{3,}\n(.*?)-{3,}\n(.*)"
    result = re.search(pattern, string_template, re.DOTALL)
    if not result:
        return string_template
    return result.group(2)


def load_template(
    template: str,
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

    recursive_paths = [TEMPLATES_PATH] + [p for p in Path(TEMPLATES_PATH).rglob("*") if p.is_dir()]
    loader = FileSystemLoader(searchpath=recursive_paths)

    undefined_vars = []

    class KeepUndefined(Undefined):
        def __str__(self):
            # Keep a list of variables that have not yet been filled.
            undefined_vars.append(self._undefined_name)

            # We leave double braces here, so that json instructions and
            # examples etc are not misinterpreted as variables.
            return "{{" + str(self._undefined_name) + "}}"

    # Create a sandbox environment for template processing
    sandbox_env = sandbox.SandboxedEnvironment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=KeepUndefined,  # Retain unfilled placeholders
        keep_trailing_newline=False,
    )

    # Add a custom filter for comprehensive whitespace stripping
    def strip_all(s):
        if isinstance(s, str):
            return s.strip()
        return s
        
    sandbox_env.filters['strip_all'] = strip_all
    
    # Load main template
    try:
        tpl = sandbox_env.get_template(f"{template}.jinja2")
    except Exception as err:
        raise FatalError(f"Template {template} not loaded.") from err

    # Create a combined variables dict with trusted parameters taking precedence
    all_vars = {**parameters, **untrusted_inputs}

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

    messages = parse_chat(prompty, valid_roles=["system", "user", "assistant", "placeholder", "developer", "human"])

    # Convert to LLMMessage
    for message in messages:
        template_content = re.sub(r"[^\w\d_]+", "", message["content"]).lower()
        if not template_content:
            # don't add empty messages
            continue
        match message["role"].lower():
            case "developer" | "system":
                output.append(SystemMessage(content=message["content"]))
            case "user" | "human":
                output.append(UserMessage(content=message["content"], source="template"))
            case "assistant":
                output.append(AssistantMessage(content=message["content"], source="template"))
            case "placeholder":
                # Remove everything except word chars to get the variable name
                if template_content == "context":
                    # special case for context
                    output.extend(context)
                elif template_content == "records":
                    output.extend([rec.as_message() for rec in records if isinstance(rec, Record)])
                else:
                    err = (f"Missing {template_content} in placeholder vars.",)
                    raise ProcessingError(err)
            case _:
                err = (f"Missing {template_content} in placeholder vars.",)
                if fail_on_missing_placeholders:
                    raise ProcessingError(err)
                logger.warning(err)

    return output
