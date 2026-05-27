"""Native Pydantic v2 message types for Buttermilk.

This module provides Buttermilk's own message, function-call, and LLM-response
types. Every class preserves the same field names, types, and access patterns
used throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Required, TypedDict

# ---------------------------------------------------------------------------
# Function-call types
# ---------------------------------------------------------------------------


@dataclass
class FunctionCall:
    """A single tool/function call requested by the model.

    This is intentionally a *dataclass* (not a Pydantic model) — it's compared
    by identity in isinstance checks and used inside union fields of Pydantic models.
    """

    id: str
    """Unique id for this call (provider-assigned)."""

    arguments: str
    """JSON-encoded arguments string."""

    name: str
    """Name of the function to call."""


# ---------------------------------------------------------------------------
# Chat message types
# ---------------------------------------------------------------------------


class SystemMessage(BaseModel):
    """System/developer message containing instructions for the model."""

    content: str
    """The content of the message."""

    type: Literal["SystemMessage"] = "SystemMessage"


class UserMessage(BaseModel):
    """User message — input from end users or a catch-all for data provided to the model."""

    content: Union[str, list[Union[str]]]
    """The content of the message (str or list of str for multimodal)."""

    source: str
    """The name of the agent that sent this message."""

    type: Literal["UserMessage"] = "UserMessage"


class AssistantMessage(BaseModel):
    """Assistant message sampled from the language model."""

    content: Union[str, list[FunctionCall]]
    """The content of the message."""

    thought: str | None = None
    """Optional reasoning text (used by reasoning models)."""

    source: str
    """The name of the agent that sent this message."""

    type: Literal["AssistantMessage"] = "AssistantMessage"

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Function execution result types
# ---------------------------------------------------------------------------


class FunctionExecutionResult(BaseModel):
    """Result of executing a single function/tool call."""

    content: str
    """The output of the function call."""

    name: str
    """The name of the function that was called."""

    call_id: str
    """The ID of the function call."""

    is_error: bool | None = None
    """Whether the function call resulted in an error."""


class FunctionExecutionResultMessage(BaseModel):
    """Container for multiple function execution results."""

    content: list[FunctionExecutionResult]

    type: Literal["FunctionExecutionResultMessage"] = "FunctionExecutionResultMessage"


# ---------------------------------------------------------------------------
# LLMMessage union (discriminated on ``type``)
# ---------------------------------------------------------------------------

LLMMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, FunctionExecutionResultMessage],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage / result types
# ---------------------------------------------------------------------------


@dataclass
class RequestUsage:
    """Token usage for a single LLM request."""

    prompt_tokens: int
    completion_tokens: int


FinishReasons = Literal["stop", "length", "function_calls", "content_filter", "unknown"]


@dataclass
class TopLogprob:
    logprob: float
    bytes: list[int] | None = None


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: list[TopLogprob] | None = None
    bytes: list[int] | None = None


class CreateResult(BaseModel):
    """Result of a model completion."""

    finish_reason: FinishReasons
    """The reason the model finished generating the completion."""

    content: Union[str, list[FunctionCall]]
    """The output of the model completion."""

    usage: RequestUsage
    """Token usage for prompt and completion."""

    cached: bool
    """Whether the completion was generated from a cached response."""

    logprobs: list[ChatCompletionTokenLogprob] | None = None
    """Log probabilities of the tokens, if available."""

    thought: str | None = None
    """Reasoning text for reasoning models, if available."""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# ModelInfo — TypedDict, accessed via .get() in existing code
# ---------------------------------------------------------------------------


class ModelFamily:
    """Namespace for model family constants.

    Provides constants like ``ModelFamily.GPT_4O`` for model family identification.
    """

    GPT_5 = "gpt-5"
    GPT_41 = "gpt-41"
    GPT_45 = "gpt-45"
    GPT_4O = "gpt-4o"
    O1 = "o1"
    O3 = "o3"
    O4 = "o4"
    GPT_4 = "gpt-4"
    GPT_35 = "gpt-35"
    R1 = "r1"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_4_OPUS = "claude-4-opus"
    CLAUDE_4_SONNET = "claude-4-sonnet"
    LLAMA_3_3_8B = "llama-3.3-8b"
    LLAMA_3_3_70B = "llama-3.3-70b"
    LLAMA_4_SCOUT = "llama-4-scout"
    LLAMA_4_MAVERICK = "llama-4-maverick"
    CODESRAL = "codestral"
    OPEN_CODESRAL_MAMBA = "open-codestral-mamba"
    MISTRAL = "mistral"
    MINISTRAL = "ministral"
    PIXTRAL = "pixtral"
    UNKNOWN = "unknown"

    @staticmethod
    def is_claude(family: str) -> bool:
        return family in (
            ModelFamily.CLAUDE_3_HAIKU,
            ModelFamily.CLAUDE_3_SONNET,
            ModelFamily.CLAUDE_3_OPUS,
            ModelFamily.CLAUDE_3_5_HAIKU,
            ModelFamily.CLAUDE_3_5_SONNET,
            ModelFamily.CLAUDE_3_7_SONNET,
            ModelFamily.CLAUDE_4_OPUS,
            ModelFamily.CLAUDE_4_SONNET,
        )

    @staticmethod
    def is_gemini(family: str) -> bool:
        return family in (
            ModelFamily.GEMINI_1_5_FLASH,
            ModelFamily.GEMINI_1_5_PRO,
            ModelFamily.GEMINI_2_0_FLASH,
            ModelFamily.GEMINI_2_5_PRO,
            ModelFamily.GEMINI_2_5_FLASH,
        )

    @staticmethod
    def is_openai(family: str) -> bool:
        return family in (
            ModelFamily.GPT_5,
            ModelFamily.GPT_45,
            ModelFamily.GPT_41,
            ModelFamily.GPT_4O,
            ModelFamily.O1,
            ModelFamily.O3,
            ModelFamily.O4,
            ModelFamily.GPT_4,
            ModelFamily.GPT_35,
        )

    @staticmethod
    def is_llama(family: str) -> bool:
        return family in (
            ModelFamily.LLAMA_3_3_8B,
            ModelFamily.LLAMA_3_3_70B,
            ModelFamily.LLAMA_4_SCOUT,
            ModelFamily.LLAMA_4_MAVERICK,
        )

    @staticmethod
    def is_mistral(family: str) -> bool:
        return family in (
            ModelFamily.CODESRAL,
            ModelFamily.OPEN_CODESRAL_MAMBA,
            ModelFamily.MISTRAL,
            ModelFamily.MINISTRAL,
            ModelFamily.PIXTRAL,
        )


class ModelInfo(TypedDict, total=False):
    """Model metadata dictionary — accessed via ``.get()`` in existing code.

    Model metadata dictionary — accessed via ``.get()`` in existing code.
    """

    vision: Required[bool]
    function_calling: Required[bool]
    json_output: Required[bool]
    family: Required[str]
    structured_output: Required[bool]
    multiple_system_messages: bool | None


# ---------------------------------------------------------------------------
# Convenience re-export list
# ---------------------------------------------------------------------------

__all__ = [
    "AssistantMessage",
    "ChatCompletionTokenLogprob",
    "CreateResult",
    "FinishReasons",
    "FunctionCall",
    "FunctionExecutionResult",
    "FunctionExecutionResultMessage",
    "LLMMessage",
    "ModelFamily",
    "ModelInfo",
    "RequestUsage",
    "SystemMessage",
    "TopLogprob",
    "UserMessage",
]
