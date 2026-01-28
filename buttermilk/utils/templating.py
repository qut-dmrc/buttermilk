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

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

import regex as re  # For regular expression operations, used in _parse_prompty
from autogen_core.models import (
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)  # Autogen message types
from jinja2 import (  # Jinja2 templating components
    FileSystemLoader,
    Undefined,
    sandbox,
)
from pydantic import BaseModel, PrivateAttr  # Pydantic components

from buttermilk import bm, logger  # Centralized logger
from buttermilk._core.constants import TEMPLATES_PATH  # Default path for templates
from buttermilk._core.exceptions import FatalError, ProcessingError  # Custom exceptions
from buttermilk._core.types import BaseRecord  # Core Buttermilk Record type
from buttermilk.utils.utils import (
    clean_empty_values,
    list_files,
    list_files_with_content,
)  # Utilities for file listing


def _get_template_search_paths() -> list[str]:
    """Get the search paths for templates, prioritizing session-specific paths."""
    search_paths = []
    try:
        search_paths.extend(bm.session_info.template_paths)
    except Exception as e:
        logger.warning(f"Could not get template paths from session: {e}")

    # Add default path if it's not already there (convert Path to str)
    templates_path_str = str(TEMPLATES_PATH)
    if templates_path_str not in search_paths:
        search_paths.append(templates_path_str)

    # Deduplicate and expand subdirectories
    final_paths = []
    for path in search_paths:
        path_str = str(path)  # Ensure it's a string
        if path_str not in final_paths:
            final_paths.append(path_str)
            final_paths.extend(
                [str(p) for p in Path(path_str).rglob("*") if p.is_dir()]
            )

    return final_paths


# Session-scoped cache for Jinja2 environments
# Uses WeakValueDictionary so environments are garbage collected when session ends
_session_jinja_envs: WeakValueDictionary[str, sandbox.SandboxedEnvironment] = (
    WeakValueDictionary()
)


def _get_cached_jinja_environment(
    search_paths_tuple: tuple[str, ...],
) -> sandbox.SandboxedEnvironment:
    """Get a session-scoped cached Jinja2 environment for the given search paths.

    This prevents creating new FileSystemLoader and SandboxedEnvironment instances
    for every template load, while allowing cleanup when sessions end.

    Cache is session-scoped and automatically cleaned up when the session ends
    (via WeakValueDictionary).

    Args:
        search_paths_tuple: Tuple of search paths (must be hashable for caching)

    Returns:
        Cached SandboxedEnvironment instance for this session
    """
    # Create cache key combining session_id and search paths
    try:
        session_id = bm.session_info.session_id
    except Exception:
        session_id = "global"  # Fallback for non-session contexts

    cache_key = f"{session_id}:{','.join(search_paths_tuple)}"

    # Return cached environment if it exists
    if cache_key in _session_jinja_envs:
        return _session_jinja_envs[cache_key]

    # Create new environment
    file_system_loader = FileSystemLoader(searchpath=list(search_paths_tuple))

    # Note: We can't define KeepUndefinedAndCollect here because it needs to collect
    # undefined variables per-render. We'll pass it during environment creation instead.
    sandboxed_env = sandbox.SandboxedEnvironment(
        loader=file_system_loader,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=Undefined,  # Default undefined, will be overridden per-render
        keep_trailing_newline=False,
    )

    # Custom filter to strip all leading/trailing whitespace from a string
    def strip_all_whitespace(s: Any) -> Any:
        """Jinja filter to strip whitespace if input is a string."""
        if isinstance(s, str):
            return s.strip()
        return s

    sandboxed_env.filters["strip_all"] = strip_all_whitespace

    # Cache the environment (weak reference allows GC when session ends)
    _session_jinja_envs[cache_key] = sandboxed_env

    logger.debug(
        "Created new Jinja2 environment for session",
        session_id=session_id,
        cache_key=cache_key,
    )

    return sandboxed_env


class KeyValueCollector(BaseModel):
    """A collector for key-value pairs, typically used for populating prompt templates.

    This class provides methods to add, update, set, and retrieve data.
    Values associated with a key are stored as a list, allowing multiple values
    to be collected under the same key.

    Attributes:
        _data (dict[str, list[Any]]): A private dictionary storing the
            collected key-value pairs. Values are stored as lists to accumulate
            multiple items under the same key.

    """

    _data: dict[str, list[Any]] = PrivateAttr(
        default_factory=dict
    )  # Values are always lists

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
        # Filter out empty/None values
        if value is None or value in ([], {}, "None"):
            return

        # Ensure that value is treated as a list of items to be added
        items_to_add = (
            value if isinstance(value, list) and not isinstance(value, str) else [value]
        )

        if key in self._data:
            self._data[key].extend(items_to_add)
        else:
            self._data[key] = items_to_add  # Initialize with the list of items

    def set(self, key: str, value: Any) -> None:
        """Sets or replaces the value for a given key.

        The value is stored as a list containing the single new value,
        overwriting any previous values for that key.
        Skips setting if the value is None, an empty list, an empty dict, or the string "None".

        Args:
            key (str): The key whose value is to be set.
            value (Any): The new value for the key.

        """
        if value is not None and value not in ([], {}, "None"):
            self._data[key] = [value]  # Store as a list with one item

    def get_dict(self) -> dict[str, list[Any]]:
        """Returns a copy of the internal data dictionary.

        Returns:
            dict[str, list[Any]]: A dictionary where keys are strings and values
            are lists of collected items.

        """
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> list[Any]:
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

    def __getitem__(self, key: str) -> list[Any]:
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

    def clear(self) -> None:
        """Clears all collected data, resetting the internal dictionary to empty."""
        self._data.clear()


def calculate_template_hash(template_name: str) -> tuple[str, str]:
    """Calculate SHA-256 hash of a template file for tracking template versions.

    Args:
        template_name (str): The name of the template file (without .jinja2 extension)
            to calculate hash for, located in `TEMPLATES_PATH`.

    Returns:
        tuple[str, str]: A tuple containing:
            - str: The SHA-256 hash prefixed with "sha256:" for clarity
            - str: The full path to the template file that was hashed

    Raises:
        FatalError: If the template file cannot be found or read.

    """
    template_filename = f"{template_name}.jinja2"

    # Search for the template file in the configured search paths
    search_paths = _get_template_search_paths()

    template_path = None
    for search_path in search_paths:
        potential_path = Path(search_path) / template_filename
        if potential_path.exists() and potential_path.is_file():
            template_path = potential_path
            break

    if template_path is None:
        raise FatalError(
            f"Template file '{template_filename}' not found in {search_paths} or their subdirectories."
        )

    try:
        # Calculate hash using unified hashing module
        from buttermilk._core.hashing import compute_template_hash_from_file

        hash_value = compute_template_hash_from_file(template_path)
        return hash_value, str(template_path)
    except Exception as e:
        raise FatalError(
            f"Failed to read template file '{template_path}' for hash calculation: {e!s}"
        ) from e


def get_templates(
    pattern: str = "", parent: str = "", extension: str = ".jinja2"
) -> list[tuple[str, str]]:  # Added default extension
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
        filename=pattern,  # Parameter is 'filename', not 'filename_pattern'
        parent=parent,
        extension=effective_extension,
    )
    # Strip the .jinja2 (or any extension) part for the template name
    return [
        (Path(tpl_path).stem, content) for tpl_path, content in templates_with_content
    ]


def get_template_names(
    pattern: str = "", parent: str = "", extension: str = "jinja2"
) -> list[str]:
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
        file_path.stem  # .stem gives filename without final suffix
        for file_path in list_files(
            TEMPLATES_PATH,
            filename_pattern=pattern,  # Assuming list_files takes filename_pattern
            parent_dir=parent,
            extension=extension,  # list_files might expect with or without dot
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

    Raises:
        ProcessingError: If template contains ambiguous --- horizontal rule markers
                        in the body content that could be confused with frontmatter.

    """
    # Horizontal rule pattern: --- at start of line (with optional trailing whitespace)
    horizontal_rule_pattern = r"^-{3,}\s*$"

    # Check if template starts with frontmatter (must be at very beginning)
    has_frontmatter = string_template.lstrip().startswith("---")

    if has_frontmatter:
        # Regex to find frontmatter (e.g., --- \n frontmatter \n --- \n content)
        # It captures the frontmatter in group 1 and the main content in group 2.
        # Use \A to match only at start of string (not start of any line)
        pattern = r"\A-{3,}\s*?\n(.*?)^-{3,}\s*?\n(.*)"
        match = re.search(pattern, string_template, re.DOTALL | re.MULTILINE)

        if not match:
            raise ProcessingError(
                "Template starts with --- but does not have valid frontmatter structure. "
                "Expected: ---\\nfrontmatter\\n---\\ncontent"
            )

        body_content = match.group(2)

        # Check body for additional horizontal rules
        if re.search(horizontal_rule_pattern, body_content, re.MULTILINE):
            raise ProcessingError(
                "Template contains ambiguous --- horizontal rule markers in body content "
                "after frontmatter. These could be confused with Prompty frontmatter delimiters. "
                "Please remove horizontal rules or use alternative formatting."
            )

        return body_content.strip()

    else:
        # No frontmatter - check entire template for horizontal rules
        if re.search(horizontal_rule_pattern, string_template, re.MULTILINE):
            raise ProcessingError(
                "Template contains ambiguous --- horizontal rule markers in body content. "
                "These could be confused with Prompty frontmatter delimiters. "
                "Please remove horizontal rules or use alternative formatting."
            )
        return string_template


def load_template(
    template: str,  # Name of the template file (without .jinja2 extension)
    template_vars: dict[str, Any] | None = None,  # Variables for template rendering
) -> tuple[str, set[str], str]:
    """Renders a Jinja2 template with hierarchical includes.

    Uses a sandboxed Jinja2 environment. Undefined variables in the template
    are preserved as `{{ variable_name }}` in the output, and their names are
    collected.

    The template loader searches recursively within configured template paths.

    Args:
        template (str): The name of the template file (without the .jinja2 extension)
            to load from configured template paths.
        template_vars (dict[str, Any] | None): Variables available to the template.
            Can control template logic, includes, and content substitution.
            Empty values (None, "", [], {}) are automatically removed to enforce
            fail-fast - variables with empty values will be treated as missing/unfilled.

    Returns:
        tuple[str, set[str], str]: A tuple containing:
            - str: The fully rendered template content as a string.
            - set[str]: A set of variable names present in template but not in template_vars.
            - str: The SHA-256 hash of the template file content, prefixed with "sha256:".

    Raises:
        FatalError: If the specified template file cannot be loaded.

    """
    effective_vars = template_vars or {}

    # Clean empty values to enforce fail-fast
    # Empty values (None, "", [], {}) are removed so they're treated as missing
    # This prevents silent failures where empty data is rendered as valid input
    effective_vars = clean_empty_values(effective_vars)

    # Define search paths for templates using the new helper
    search_paths = _get_template_search_paths()

    # Get cached environment to prevent file descriptor leaks
    # Convert list to tuple for hashability in lru_cache
    sandboxed_env = _get_cached_jinja_environment(tuple(search_paths))

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

    # Override the undefined handler for this specific render
    # (The cached environment has a default Undefined, we override per-use)
    sandboxed_env.undefined = KeepUndefinedAndCollect

    template_filename = f"{template}.jinja2"
    try:
        jinja_template = sandboxed_env.get_template(template_filename)
    except Exception as err:  # Catch Jinja2 specific TemplateNotFound or general errors
        logger.error(
            f"Failed to load Jinja2 template '{template_filename}': {err!s}",
            exc_info=True,
        )
        raise FatalError(
            f"Template '{template}' (file: '{template_filename}') could not be loaded."
        ) from err

    # Exclude 'record' and 'context' from Jinja2 rendering - these are handled
    # specially by make_messages() as placeholder roles, not template variables.
    # If they're in rendering_context, Jinja2 would render them as JSON/dict
    # instead of leaving {{record}} for make_messages to process.
    placeholder_keys = {"record", "context"}
    rendering_context = {k: v for k, v in effective_vars.items() if k not in placeholder_keys}

    rendered_string = jinja_template.render(**rendering_context)

    # Calculate template hash for version tracking
    try:
        template_hash, _ = calculate_template_hash(template)
    except FatalError:
        # If hash calculation fails, re-raise as the template loading should have also failed
        logger.warning(
            f"Could not calculate hash for template '{template}' - this may indicate a template loading issue"
        )
        raise

    # Check for unfilled parameters if requested (fail-fast)
    # Exclude placeholder keys (record, context) - these are handled by make_messages, not Jinja2
    unfilled_vars = set(collected_undefined_vars) - placeholder_keys
    if effective_vars.get("fail_on_unfilled_parameters") and unfilled_vars:
        raise FatalError(
            f"Template '{template}' has unfilled parameters: {', '.join(sorted(unfilled_vars))}"
        )

    return rendered_string, unfilled_vars, template_hash


@dataclass
class TemplateRenderResult:
    """Result of rendering a template with merged variables.

    Provides a structured return type for template rendering operations,
    including the rendered content and metadata for traceability.

    Attributes:
        rendered: The fully rendered template string
        template_name: Name of the template that was rendered
        template_hash: SHA-256 hash of the template file for version tracking
        unfilled_vars: List of template variables that remained unfilled
    """

    rendered: str
    template_name: str
    template_hash: str
    unfilled_vars: list[str]


def render_template(
    template: str,
    template_vars: dict[str, Any] | None = None,
    *,
    base_template_vars: dict[str, Any] | None = None,
    fail_on_unfilled: bool = True,
) -> TemplateRenderResult:
    """Render a Jinja2 template with merged variables.

    This is the preferred high-level function for template rendering. It combines
    base_template_vars (config-time defaults) with template_vars (runtime values),
    where runtime values override config defaults.

    Use this function instead of calling load_template() directly when you need:
    - Merging of config-time and runtime template variables
    - Structured result with metadata
    - Consistent fail-on-unfilled behavior

    Args:
        template: Template name (without .jinja2 extension)
        template_vars: Runtime variables to fill template placeholders
        base_template_vars: Config-time defaults (e.g., from processor/agent config)
        fail_on_unfilled: If True, raise FatalError when unfilled vars remain
            (excluding 'record' and 'context' placeholders handled by make_messages)

    Returns:
        TemplateRenderResult with rendered string and metadata

    Raises:
        FatalError: If fail_on_unfilled=True and template has unfilled parameters,
            or if template cannot be loaded

    Example:
        >>> result = render_template(
        ...     template="judge_criteria",
        ...     template_vars={"text": record.text},
        ...     base_template_vars={"criteria": "Be concise"},
        ... )
        >>> print(result.rendered)
        >>> print(result.template_hash)
    """
    # Merge base (config-time) with runtime, runtime overrides
    merged = {**(base_template_vars or {}), **(template_vars or {})}
    filtered = clean_empty_values(merged) if merged else {}

    # Load and render template
    rendered_str, unfilled_vars, template_hash = load_template(template, filtered)

    # Exclude placeholders handled by make_messages (not Jinja2 variables)
    unfilled_vars = unfilled_vars - {"record", "context"}

    # Fail-fast on unfilled variables if requested
    # Use FatalError for consistency with load_template (config/setup error, not runtime)
    if unfilled_vars and fail_on_unfilled:
        raise FatalError(
            f"Template '{template}' has unfilled parameters: {', '.join(sorted(unfilled_vars))}"
        )

    return TemplateRenderResult(
        rendered=rendered_str,
        template_name=template,
        template_hash=template_hash,
        unfilled_vars=list(unfilled_vars),
    )


def _deduplicate_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Remove duplicate messages from a list while preserving order.

    Duplicates are identified by having the same role, content, and source.
    The first occurrence of each unique message is preserved.

    Args:
        messages: List of LLMMessage objects to deduplicate

    Returns:
        List of LLMMessage objects with duplicates removed

    """
    seen = set()
    deduplicated = []

    for msg in messages:
        # Create a hashable identifier for the message
        # Use role, content, and source (if available) as the key
        msg_key = (
            type(msg).__name__,  # Message type (SystemMessage, UserMessage, etc.)
            getattr(msg, "content", ""),
            getattr(msg, "source", None),
        )

        if msg_key not in seen:
            seen.add(msg_key)
            deduplicated.append(msg)
        else:
            logger.debug(
                "Removing duplicate message",
                message_type=type(msg).__name__,
                content=getattr(msg, "content", "")[:50],
            )

    if len(deduplicated) < len(messages):
        logger.info(
            "Removed duplicate messages", count=len(messages) - len(deduplicated)
        )

    return deduplicated


def _parse_chat_messages(
    chat_str: str, valid_roles: list[str] | None = None
) -> list[dict[str, str]]:
    """Simple chat message parser for Prompty-style format.

    Parses chat strings like:
        system: You are a helpful assistant
        user: Hello!
        assistant: Hi there!

    Args:
        chat_str: String containing chat messages with role prefixes
        valid_roles: Optional list of valid roles to accept

    Returns:
        List of dicts with 'role' and 'content' keys
    """
    if valid_roles is None:
        valid_roles = [
            "system",
            "user",
            "assistant",
            "placeholder",
            "developer",
            "human",
        ]

    messages = []
    current_role = None
    current_content = []

    # Split by lines and process
    for line in chat_str.split("\n"):
        # Check if this line starts with a role marker (role: or # role:)
        stripped = line.strip()
        role_match = None

        # Try matching with optional # prefix
        for role in valid_roles:
            if stripped.lower().startswith(f"# {role}:") or stripped.lower().startswith(
                f"{role}:"
            ):
                role_match = role
                # Extract content after the role marker
                if stripped.lower().startswith(f"# {role}:"):
                    content_start = stripped.index(":") + 1
                else:
                    content_start = stripped.index(":") + 1
                current_content_line = stripped[content_start:].strip()

                # Save previous message if exists
                if current_role is not None:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }
                    )

                # Start new message
                current_role = role
                current_content = [current_content_line] if current_content_line else []
                break

        # If no role match, add to current content
        if role_match is None and current_role is not None:
            current_content.append(line)

    # Don't forget the last message
    if current_role is not None:
        messages.append(
            {"role": current_role, "content": "\n".join(current_content).strip()}
        )

    return messages


def make_messages(  # noqa: PLR0912
    local_template: str,  # Rendered template string, potentially in Prompty format
    *,
    context: list[LLMMessage] | None = None,  # Conversation history
    record: BaseRecord | None = None,  # Optional record
) -> tuple[list[LLMMessage], set[str]]:
    """Construct a list of Autogen `LLMMessage` objects from a "Prompty" formatted string.

    This function first parses the `local_template` string to separate Prompty
    frontmatter (if any) from the main content. It then parses the main content into
    a list of message dictionaries, each specifying a role and content.

    These dictionaries are then converted into Autogen `LLMMessage` objects
    (e.g., `SystemMessage`, `UserMessage`, `AssistantMessage`). Special
    "placeholder" roles in the Prompty template are handled:
    -   A placeholder with content "context" (case-insensitive, after stripping
        non-alphanumerics) will be replaced by the messages in the `context` argument.
    -   A placeholder with content "record" will be replaced by converting the given
        `Record` into an `UserMessage` (using `record.as_message()`).

    Args:
        local_template (str): The string content of the rendered template,
            expected to be in Prompty format (frontmatter optional, then chat messages).
        context (list[LLMMessage] | None): An optional list of `LLMMessage` objects
            representing prior conversation history to be injected. Defaults to an empty list.
        record (BaseRecord | None): An optional list of `BaseRecord` objects to be
            injected. Defaults to an empty list.

    Returns:
        tuple[list[LLMMessage], set[str]]: A tuple containing:
            - list[LLMMessage]: A list of Autogen `LLMMessage` objects ready for use
              with an LLM client.
            - set[str]: A set of placeholder names that were successfully processed
              ("context", "record", etc.)

    Raises:
        ProcessingError:
            -   If `local_template` cannot be decoded as a Prompty format
                (e.g., due to issues in `_parse_prompty`).

    """
    output_messages: list[LLMMessage] = []
    processed_placeholders: set[str] = set()

    # Ensure context is a list (handle None or mutable default argument issues)
    if context is None:
        context = []
    elif not isinstance(context, list):
        # If context is not a list, wrap it
        context = [context]

    try:
        # Parse main content from Prompty string (strips frontmatter)
        prompty_content_str = _parse_prompty(local_template)
    except Exception as e:  # Broad catch if _parse_prompty itself fails
        err_msg = (
            f"Unable to decode template string expecting Prompty format. Error: {e!s}"
        )
        raise ProcessingError(err_msg) from e

    # Parse chat messages using our own parser (no promptflow dependency)
    parsed_chat_messages = _parse_chat_messages(
        prompty_content_str,
        valid_roles=[
            "system",
            "user",
            "assistant",
            "placeholder",
            "developer",
            "human",
        ],
    )

    # Convert parsed message dictionaries to LLMMessage objects
    for msg_dict in parsed_chat_messages:
        role_lower = msg_dict.get("role", "").lower()
        content_str = msg_dict.get("content", "")

        # Normalize content for placeholder matching: lowercase, alphanumeric only
        normalized_placeholder_key = re.sub(r"[^\w\d_]+", "", content_str).lower()

        if (
            not content_str and role_lower != "placeholder"
        ):  # Skip empty non-placeholder messages
            logger.debug("Skipping message with empty content", role=role_lower)
            continue

        if role_lower in ("developer", "system"):
            output_messages.append(SystemMessage(content=content_str))
        elif role_lower in ("user", "human"):
            output_messages.append(
                UserMessage(content=content_str, source="template_user")
            )  # Add source
        elif role_lower == "assistant":
            output_messages.append(
                AssistantMessage(content=content_str, source="template_assistant")
            )  # Add source
        elif role_lower == "placeholder":
            if normalized_placeholder_key == "context" and context:
                output_messages.extend(context)
                processed_placeholders.add("context")

            elif normalized_placeholder_key == "record" and record:
                output_messages.append(record.as_message())
                processed_placeholders.add("record")
            elif (
                content_str.strip()
            ):  # Non-empty placeholder content that's not context/record
                # Treat as user message - this handles templates where "placeholder:"
                # is used as a marker with rendered Jinja variables
                output_messages.append(
                    UserMessage(content=content_str, source="template_placeholder")
                )
            # else: empty placeholder, skip it
        else:  # Unrecognized role
            raise ProcessingError(
                f"Unrecognized role '{msg_dict.get('role')}' in Prompty template message."
            )

    # Fail-fast: Empty messages list indicates template format issue
    # This catches missing role markers (system:, user:, etc.) early
    # rather than causing cryptic "list index out of range" errors downstream
    if not output_messages:
        raise ProcessingError(
            "Template produced no messages. Ensure your template includes role markers "
            "(e.g., 'system:', 'user:', 'assistant:') at the start of message sections. "
            "Example format:\n"
            "  system:\n"
            "  You are a helpful assistant.\n"
            "  \n"
            "  user:\n"
            "  {{ user_input }}"
        )

    # Deduplicate messages before returning
    return _deduplicate_messages(output_messages), processed_placeholders
