"""Utilities for loading and rendering prompt templates, and constructing message lists.

This module provides functionalities for:
- Collecting key-value pairs for template injection (`KeyValueCollector`).
- Discovering available prompt templates (`get_templates`, `get_template_names`).
- Loading Jinja2 templates from the filesystem within a sandboxed environment,
  handling undefined variables gracefully (`load_template`).
- Parsing "Prompty" formatted strings (text files with frontmatter and chat message
  sections) and converting them into a list of Autogen `LLMMessage` objects,
  injecting context and records into specified placeholders (`_parse_prompty`,
  `make_messages`).
"""
from pathlib import Path
from typing import Any, Mapping, Sequence # For type hinting

import jmespath # For JMESPath queries, used in _resolve_mappings
import regex as re # For regular expression operations, used in _parse_prompty
from jinja2 import ( # Jinja2 templating components
    FileSystemLoader,
    Undefined,
    sandbox,
)
from pydantic import BaseModel, PrivateAttr # Pydantic components

from buttermilk import logger # Centralized logger
from autogen_core.models import AssistantMessage, UserMessage, LLMMessage, SystemMessage # Autogen message types
from buttermilk._core.defaults import TEMPLATES_PATH # Default path for templates
from buttermilk._core.exceptions import FatalError, ProcessingError # Custom exceptions
from buttermilk._core.types import Record # Core Buttermilk Record type
from buttermilk.utils.utils import list_files, list_files_with_content # Utilities for file listing


class KeyValueCollector(BaseModel):
    """A collector for key-value pairs, typically used for populating prompt templates.

    This class provides methods to add, update, set, and retrieve data.
    Values associated with a key are stored as a list, allowing multiple values
    to be collected under the same key.

    The `_resolve_mappings` and `_resolve_simple_path` methods suggest capabilities
    for resolving JMESPath expressions against the collected data, though their
    current usage within the broader Buttermilk framework might be evolving or deprecated
    as indicated by a TODO note in the original code.

    Attributes:
        _data (dict[str, Any | list[Any]]): A private dictionary storing the
            collected key-value pairs. Values are often lists to accumulate multiple
            items under the same key.
    """

    _data: dict[str, list[Any]] = PrivateAttr(default_factory=dict) # Values are always lists

    def update(self, incoming: dict[str, Any]) -> None:
        """Updates the collector with key-value pairs from an incoming dictionary.
        
        Calls `self.add` for each item, ensuring values are appended to lists.

        Args:
            incoming (dict[str, Any]): A dictionary of items to add.
        """
        for key, value in incoming.items():
            self.add(key, value)

    def add(self, key: str, value: Any) -> None:
        """Adds a value to a given key. If the key exists, appends to its list of values.

        If the provided `value` is not already a list (and not a string, as strings
        are sequences but usually treated as single values here), it's wrapped in a list
        before being added.

        Args:
            key (str): The key under which to store the value.
            value (Any): The value to add. Can be a single item or a list of items.
        """
        # Ensure that value is treated as a list of items to be added
        items_to_add = value if isinstance(value, list) and not isinstance(value, str) else [value]
        
        if key in self._data:
            self._data[key].extend(items_to_add)
        else:
            self._data[key] = items_to_add # Initialize with the list of items

    def set(self, key: str, value: Any) -> None:
        """Sets or replaces the value for a given key.
        
        The value is stored as a list containing the single new value,
        overwriting any previous values for that key.
        Skips setting if the value is None, an empty list, an empty dict, or the string "None".

        Args:
            key (str): The key whose value is to be set.
            value (Any): The new value for the key.
        """
        if value is not None and value != [] and value != {} and value != "None":
            self._data[key] = [value] # Store as a list with one item

    def get_dict(self) -> dict[str, list[Any]]: # Return type updated
        """Returns a copy of the internal data dictionary.

        Returns:
            dict[str, list[Any]]: A dictionary where keys are strings and values
            are lists of collected items.
        """
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> list[Any] | Any: # Return type updated
        """Retrieves the list of values for a key, or a default if the key is not found.

        Args:
            key (str): The key whose values are to be retrieved.
            default (Any): The value to return if the key is not found.
                           Defaults to None.

        Returns:
            list[Any] | Any: The list of values associated with the key, or the
            `default` value if the key does not exist.
        """
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> list[Any]: # Return type updated
        """Allows dictionary-style access to the collected values for a key.

        Args:
            key (str): The key whose values are to be retrieved.

        Returns:
            list[Any]: The list of values associated with the key.

        Raises:
            KeyError: If the key is not found.
        """
        return self._data[key]

    def init(self, keys: list[str]) -> None:
        """Initializes specified keys in the data dictionary with empty lists.

        Useful for ensuring certain keys exist before attempting to add values to them.

        Args:
            keys (list[str]): A list of key names to initialize.
        """
        for key in keys:
            self._data[key] = []

    # The docstring for the section below was not part of the class definition.
    # It seems to be a general comment about routing variables.
    # """Routes variables between workflow steps using mappings
    # Data is essentially a dict of lists, where each key is the name of a step and
    # each list is the output of an agent in a step.
    # """

    def _resolve_mappings(self, mappings: dict[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        """Resolves JMESPath expressions defined in `mappings` against `data`. (DEPRECATED or under review)

        This method recursively processes a `mappings` dictionary. For each target
        key in `mappings`, its corresponding source specification (a JMESPath string,
        a list of JMESPath strings for aggregation, or a nested mapping) is resolved
        against the `data` dictionary.

        Note:
            A TODO comment in the original code suggests this method might be
            deprecated or no longer needed. Its usage should be reviewed.

        Args:
            mappings (dict[str, Any]): A dictionary where keys are target variable
                names and values are JMESPath expressions (str), lists of
                JMESPath expressions, or nested mapping dictionaries.
            data (Mapping[str, Any]): The data structure (e.g., from a Pydantic model's
                `model_dump()`) to query with JMESPath.

        Returns:
            dict[str, Any]: A dictionary where keys are the target variable names
            from `mappings` and values are the results of their resolved JMESPath
            queries against `data`. Empty values or containers are removed.
        """
        logger.warning("_resolve_mappings is under review and might be deprecated. Current caller should be checked.")
        resolved: dict[str, Any] = {}

        data_dict = data.model_dump() if hasattr(data, "model_dump") else dict(data) # Ensure it's a dict

        if isinstance(mappings, str): # Base case for recursion: a single JMESPath string
            # This path seems unlikely given the top-level 'mappings' type hint is dict.
            # It implies this function might be called recursively with a string mapping.
            # However, the loop below processes dict items. This branch needs clarification.
            return self._resolve_simple_path(mappings, data_dict) or {} # Ensure dict return

        for target_key, source_spec in mappings.items():
            if isinstance(source_spec, Sequence) and not isinstance(source_spec, str): # List of paths for aggregation
                aggregated_results = []
                for src_path in source_spec:
                    # Recursive call for each path in the list
                    result = self._resolve_mappings(src_path, data_dict) # type: ignore # src_path is str, expects dict
                    if result: # Append if result is not empty/None
                        if isinstance(result, list):
                            aggregated_results.extend(result)
                        else:
                            aggregated_results.append(result)
                if aggregated_results: resolved[target_key] = aggregated_results
            elif isinstance(source_spec, Mapping): # Nested mapping dictionary
                resolved[target_key] = self._resolve_mappings(source_spec, data_dict) # type: ignore # source_spec is Mapping
            else: # Simple JMESPath string
                resolved_value = self._resolve_simple_path(str(source_spec), data_dict)
                if resolved_value is not None: resolved[target_key] = resolved_value
        
        # Remove keys where value is None, or an empty dict/list
        return {k: v for k, v in resolved.items() if v is not None and v != {} and v != []}

    def _resolve_simple_path(self, path: str, data: Mapping[str, Any]) -> Any:
        """Resolves a single JMESPath expression against the provided data. (DEPRECATED or under review)

        Args:
            path (str): The JMESPath expression string.
            data (Mapping[str, Any]): The data structure to query.

        Returns:
            Any: The result of the JMESPath search, or `None` if the path is empty,
                 invalid, or yields no result.
        """
        if not path:
            return None
        try:
            return jmespath.search(path, data)
        except jmespath.exceptions.JMESPathError as e: # Catch specific JMESPath errors
            logger.warning(f"JMESPath search failed for path '{path}': {e!s}. Returning None.")
            return None
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during JMESPath search for path '{path}': {e!s}", exc_info=True)
            return None


def get_templates(pattern: str = "", parent: str = "", extension: str = ".jinja2") -> list[tuple[str, str]]: # Added default extension
    """Lists template files and their content from the configured `TEMPLATES_PATH`.

    Args:
        pattern (str): A glob pattern to filter template filenames (e.g., "user_*").
                       Defaults to "" (match all).
        parent (str): A subdirectory within `TEMPLATES_PATH` to search in.
                      Defaults to "" (search in `TEMPLATES_PATH` root).
        extension (str): The file extension for templates (including the dot).
                         Defaults to ".jinja2".

    Returns:
        list[tuple[str, str]]: A list of tuples, where each tuple contains:
            - The template name (filename without the .jinja2 extension).
            - The content of the template file as a string.
    """
    # Ensure extension starts with a dot, or add it if only chars are provided.
    effective_extension = extension
    if extension and not extension.startswith("."):
        effective_extension = "." + extension
    
    templates_with_content = list_files_with_content(
        TEMPLATES_PATH,
        filename_pattern=pattern, # Assuming list_files_with_content takes filename_pattern
        parent_dir=parent,
        extension=effective_extension,
    )
    # Strip the .jinja2 (or any extension) part for the template name
    return [(Path(tpl_path).stem, content) for tpl_path, content in templates_with_content]


def get_template_names(pattern: str = "", parent: str = "", extension: str = "jinja2") -> list[str]:
    """Lists the names of template files found in the configured `TEMPLATES_PATH`.

    Args:
        pattern (str): A glob pattern to filter template filenames. Defaults to "".
        parent (str): A subdirectory within `TEMPLATES_PATH` to search. Defaults to "".
        extension (str): The file extension for templates (excluding the dot).
                         Defaults to "jinja2".

    Returns:
        list[str]: A list of template names (filenames without the extension).
    """
    return [
        file_path.stem # .stem gives filename without final suffix
        for file_path in list_files(
            TEMPLATES_PATH,
            filename_pattern=pattern, # Assuming list_files takes filename_pattern
            parent_dir=parent,
            extension=extension, # list_files might expect with or without dot
        )
    ]


def _parse_prompty(string_template: str) -> str:
    """Parses a "Prompty" formatted string to extract the main content.

    A Prompty file typically has a YAML/JSON frontmatter section enclosed in
    triple-dashed lines (---). This function extracts the content *after*
    the frontmatter. If no frontmatter is detected, it returns the original string.

    Args:
        string_template (str): The string content of a Prompty template.

    Returns:
        str: The main content of the Prompty template (after the frontmatter),
             or the original string if no frontmatter is found.
    """
    # Regex to find frontmatter (e.g., --- \n frontmatter \n --- \n content)
    # It captures the frontmatter in group 1 and the main content in group 2.
    pattern = r"^-{3,}\s*?\n(.*?)^-{3,}\s*?\n(.*)"
    match = re.search(pattern, string_template, re.DOTALL | re.MULTILINE)
    if not match:
        return string_template # No frontmatter found, return original string
    return match.group(2).strip() # Return the content part, stripped


def load_template(
    template: str, # Name of the template file (without .jinja2 extension)
    parameters: dict[str, Any], # Parameters for template rendering (trusted)
    untrusted_inputs: dict[str, Any] | None = None, # User inputs (less trusted)
) -> tuple[str, set[str]]:
    """Renders a Jinja2 template with hierarchical includes and security considerations.

    It uses a sandboxed Jinja2 environment to limit potential risks from templates
    that might include user-provided data. Undefined variables in the template
    are preserved as `{{ variable_name }}` in the output, and their names are
    collected.

    The template loader searches recursively within `TEMPLATES_PATH`.

    Args:
        template (str): The name of the template file (without the .jinja2 extension)
            to load from the `TEMPLATES_PATH`.
        parameters (dict[str, Any]): A dictionary of "trusted" parameters that
            are directly available to the template. These can control template
            logic, includes, etc.
        untrusted_inputs (dict[str, Any] | None): Optional. A dictionary of "untrusted"
            user-provided inputs. These are also made available to the template
            but might be treated with more caution or subjected to stricter escaping
            if the sandbox environment were configured for autoescaping (currently not).
            Defaults to an empty dictionary if None.

    Returns:
        tuple[str, set[str]]: A tuple containing:
            - str: The fully rendered template content as a string.
            - set[str]: A set of strings, where each string is the name of a
              variable that was present in the template but not found in
              `parameters` or `untrusted_inputs`.

    Raises:
        FatalError: If the specified template file cannot be loaded.
    """
    effective_untrusted_inputs = untrusted_inputs or {}
    
    # Define search paths for templates: TEMPLATES_PATH and all its subdirectories
    recursive_search_paths = [TEMPLATES_PATH] + [p for p in Path(TEMPLATES_PATH).rglob("*") if p.is_dir()]
    file_system_loader = FileSystemLoader(searchpath=recursive_search_paths)

    collected_undefined_vars: list[str] = []

    class KeepUndefinedAndCollect(Undefined):
        """Custom Undefined type to keep undefined variables in the template
        and collect their names.
        """
        def __str__(self) -> str:
            # Add the undefined variable name to our list
            collected_undefined_vars.append(self._undefined_name)
            # Render as {{ variable_name }} to make it clear it was undefined
            return "{{" + str(self._undefined_name) + "}}"

    # Create a sandboxed Jinja2 environment
    sandboxed_env = sandbox.SandboxedEnvironment(
        loader=file_system_loader,
        trim_blocks=True, # Removes first newline after a block
        lstrip_blocks=True, # Strips leading whitespace from line to block
        undefined=KeepUndefinedAndCollect, # Custom handler for undefined variables
        keep_trailing_newline=False, # Removes newline at the end of the template output
    )

    # Custom filter to strip all leading/trailing whitespace from a string
    def strip_all_whitespace(s: Any) -> Any:
        """Jinja filter to strip whitespace if input is a string."""
        if isinstance(s, str):
            return s.strip()
        return s # Return non-strings as is
        
    sandboxed_env.filters['strip_all'] = strip_all_whitespace
    
    template_filename = f"{template}.jinja2"
    try:
        jinja_template = sandboxed_env.get_template(template_filename)
    except Exception as err: # Catch Jinja2 specific TemplateNotFound or general errors
        logger.error(f"Failed to load Jinja2 template '{template_filename}': {err!s}", exc_info=True)
        raise FatalError(f"Template '{template}' (file: '{template_filename}') could not be loaded.") from err

    # Combine parameters and untrusted inputs for rendering context.
    # Trusted `parameters` can override `untrusted_inputs` if keys collide.
    rendering_context = {**effective_untrusted_inputs, **parameters}

    rendered_string = jinja_template.render(**rendering_context)

    return rendered_string, set(collected_undefined_vars)


def make_messages(
    local_template: str, # Rendered template string, potentially in Prompty format
    context: list[LLMMessage] | None = None, # Optional conversation history
    records: list[Record] | None = None,   # Optional list of records
    fail_on_missing_placeholders: bool = False
) -> list[LLMMessage]:
    """Constructs a list of Autogen `LLMMessage` objects from a "Prompty" formatted string.

    This function first parses the `local_template` string to separate Prompty
    frontmatter (if any) from the main content. It then uses PromptFlow's utility
    (`promptflow.core._prompty_utils.parse_chat`) to parse the main content into
    a list of message dictionaries, each specifying a role and content.

    These dictionaries are then converted into Autogen `LLMMessage` objects
    (e.g., `SystemMessage`, `UserMessage`, `AssistantMessage`). Special
    "placeholder" roles in the Prompty template are handled:
    -   A placeholder with content "context" (case-insensitive, after stripping
        non-alphanumerics) will be replaced by the messages in the `context` argument.
    -   A placeholder with content "records" will be replaced by converting each
        `Record` in the `records` argument into an `UserMessage` (using `record.as_message()`).
    -   Other placeholder contents will raise a `ProcessingError` if
        `fail_on_missing_placeholders` is True, or log a warning otherwise.

    Args:
        local_template (str): The string content of the rendered template,
            expected to be in Prompty format (frontmatter optional, then chat messages).
        context (list[LLMMessage] | None): An optional list of `LLMMessage` objects
            representing prior conversation history to be injected. Defaults to an empty list.
        records (list[Record] | None): An optional list of `Record` objects to be
            injected. Defaults to an empty list.
        fail_on_missing_placeholders (bool): If True, raises a `ProcessingError`
            when a placeholder (other than "context" or "records") is encountered
            in the Prompty template that cannot be filled. If False (default),
            a warning is logged, and the placeholder might be ignored or result
            in missing content.

    Returns:
        list[LLMMessage]: A list of Autogen `LLMMessage` objects ready for use
        with an LLM client.

    Raises:
        ValueError: If `local_template` cannot be decoded as a Prompty format
            (e.g., due to issues in `_parse_prompty`).
        ProcessingError: If `fail_on_missing_placeholders` is True and an unknown
            placeholder is encountered.
    """
    output_messages: list[LLMMessage] = []
    active_context = context or [] # Ensure list
    active_records = records or [] # Ensure list

    try:
        # Parse main content from Prompty string (strips frontmatter)
        prompty_content_str = _parse_prompty(local_template)
    except Exception as e: # Broad catch if _parse_prompty itself fails
        err_msg = f"Unable to decode template string expecting Prompty format. Error: {e!s}"
        logger.error(err_msg, exc_info=True)
        raise ValueError(err_msg) from e

    # Use PromptFlow's utility to parse chat messages from the Prompty content
    from promptflow.core._prompty_utils import parse_chat # Local import as it's a specific utility

    parsed_chat_messages = parse_chat(
        prompty_content_str, 
        valid_roles=["system", "user", "assistant", "placeholder", "developer", "human"] # Allowed roles in Prompty
    )

    # Convert parsed message dictionaries to LLMMessage objects
    for msg_dict in parsed_chat_messages:
        role_lower = msg_dict.get("role", "").lower()
        content_str = msg_dict.get("content", "")
        
        # Normalize content for placeholder matching: lowercase, alphanumeric only
        normalized_placeholder_key = re.sub(r"[^\w\d_]+", "", content_str).lower()

        if not content_str and role_lower != "placeholder": # Skip empty non-placeholder messages
            logger.debug(f"Skipping message with empty content for role '{role_lower}'.")
            continue

        if role_lower in ("developer", "system"):
            output_messages.append(SystemMessage(content=content_str))
        elif role_lower in ("user", "human"):
            output_messages.append(UserMessage(content=content_str, source="template_user")) # Add source
        elif role_lower == "assistant":
            output_messages.append(AssistantMessage(content=content_str, source="template_assistant")) # Add source
        elif role_lower == "placeholder":
            if normalized_placeholder_key == "context":
                output_messages.extend(active_context)
            elif normalized_placeholder_key == "records":
                output_messages.extend([rec.as_message() for rec in active_records if isinstance(rec, Record)])
            else: # Unknown placeholder
                err_msg = f"Unknown placeholder '{{{{{content_str}}}}}' (normalized: '{normalized_placeholder_key}') in Prompty template."
                if fail_on_missing_placeholders:
                    logger.error(err_msg)
                    raise ProcessingError(err_msg)
                logger.warning(f"{err_msg} Placeholder will be ignored.")
        else: # Unrecognized role
            logger.warning(f"Unrecognized role '{msg_dict.get('role')}' in Prompty template message. Content: '{content_str[:100]}...' Message ignored.")

    return output_messages
