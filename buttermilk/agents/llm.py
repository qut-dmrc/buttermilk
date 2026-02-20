"""LLM-based Agent for Buttermilk.

This module provides the `LLMAgent` class, a foundational agent that integrates
with large language models (LLMs) to process tasks. It extends the base `Agent`
class with the ability to:
1. Fill prompt templates with input data and conversation context
2. Call LLMs with the rendered prompts
3. Parse structured outputs from LLM responses

`LLMAgent` serves as a base class for more specialized agents that require
LLM interaction, handling the core template rendering and LLM communication
workflow.
"""

from typing import Any

import pydantic
from autogen_core import CancellationToken

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llm_core import LLMCore


class LLMAgent(Agent):
    """Agent that uses an LLM for text processing and generation.

        `LLMAgent` extends the base `Agent` class to add LLM-powered capabilities.
        It manages the complete workflow of:
        1. Loading and rendering prompt templates with provided data
        2. Communicating with LLMs through the global LLM manager
        3. Parsing and validating LLM responses

        The agent can work with both unstructured text responses and structured
        outputs (when `output_model` is specified as a Pydantic model).

        Configuration (from `AgentConfig`):
            template: Name of the prompt template to use (required in parameters)
            model: Name of the LLM model to use (e.g., 'gpt-4', 'claude-3')
            temperature: LLM temperature parameter for response variability
            fail_on_unfilled_parameters: Whether to fail if template variables are missing
            tools: List of tools (functions) the agent can use
            output_model (type[pydantic.BaseModel] | None): Pydantic model
                for structured output parsing.

    Attributes:
            _model (str): The name/identifier of the LLM model this agent uses.
            _tools_list (list[Tool]): List of Autogen-compatible tool objects.
            _fail_on_unfilled_parameters (bool): If True, raises an error when
                template variables are missing from inputs.

    Example:
            ```python
            agent = LLMAgent(
                agent_name="Analyzer",
                role="TEXT_ANALYZER",
                parameters={"template": "analysis_prompt", "model": "gpt-4", "fail_on_unfilled_parameters": True},
            )

            result = await agent.invoke(
                AgentInput(inputs={"text": "Hello world"})
            )
            ```

    """

    def __init__(self, *, output_model: type[pydantic.BaseModel] = None, **kwargs: Any) -> None:
        """Initialize an LLMAgent with the provided configuration.

        Extracts the model name from parameters and stores it in `_model`.
        The actual model initialization happens in `init_model()`.

        Args:
            **kwargs: Configuration parameters passed to AgentConfig.
                Must include 'model' and 'template' in parameters.

        Raises:
            ValueError: If 'model' and 'template' is not specified in parameters.

        """
        if "name_components" not in kwargs:
            kwargs["name_components"] = ["role", "model", "unique_identifier"]
        super().__init__(**kwargs)
        if "model" not in self.parameters:
            raise ValueError(f"Agent {self.agent_name}: 'model' is required in agent parameters.")
        if "template" not in self.parameters:
            raise ValueError(f"Agent {self.agent_name}: 'template' is required in agent parameters.")

        # Initialize private attributes
        self.output_model: type[pydantic.BaseModel] = output_model or None

        # Initialize the shared LLM core
        # LLMCore.template_vars receives all non-LLM-config parameters for template rendering
        self.llm_core = LLMCore(
            model=self.parameters["model"],
            template=self.parameters["template"],
            output_model=output_model,
            tools=self._tools or [],
            fail_on_unfilled_parameters=self.parameters.get("fail_on_unfilled_parameters", True),
            # LLM inference parameters (optional)
            temperature=self.parameters.get("temperature"),
            max_tokens=self.parameters.get("max_tokens"),
            top_p=self.parameters.get("top_p"),
            top_k=self.parameters.get("top_k"),
            frequency_penalty=self.parameters.get("frequency_penalty"),
            presence_penalty=self.parameters.get("presence_penalty"),
            stop_sequences=self.parameters.get("stop_sequences"),
            seed=self.parameters.get("seed"),
            # Template variables (criteria, instructions, etc.) - everything else
            template_vars=self._extract_template_vars(),
        )

    def _extract_template_vars(self) -> dict[str, Any]:
        """Extract template variables from parameters (everything not LLM config)."""
        llm_config_keys = {
            "model",
            "template",
            "output_model",
            "tools",
            "fail_on_unfilled_parameters",
            "output_col",
            "human_in_loop",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
        }
        return {k: v for k, v in self.parameters.items() if k not in llm_config_keys}

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AgentOutput:
        """Core processing logic: uses LLMCore to process and wraps result in AgentOutput.

        Args:
            message: The `AgentInput` message containing data, context, and
                parameters for this processing step.
            cancellation_token: Optional. A token to signal cancellation of the
                LLM call or other async operations.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput: An `AgentOutput` message with the processed LLM response.

        Raises:
            ProcessingError: If LLM processing fails.
        """
        logger.debug(f"Agent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}.")

        # Use existing LLM core or create new one with runtime parameter overrides
        if message.parameters:
            # Create a new LLMCore instance with runtime parameters
            merged_params = {**self.parameters, **message.parameters}
            llm_core = LLMCore(
                model=merged_params.get("model", self.parameters["model"]),
                template=self.parameters["template"],  # Template is fixed at init
                output_model=self.output_model,
                tools=self._tools or [],
                fail_on_unfilled_parameters=merged_params.get("fail_on_unfilled_parameters", True),
                temperature=merged_params.get("temperature"),
                max_tokens=merged_params.get("max_tokens"),
                top_p=merged_params.get("top_p"),
                top_k=merged_params.get("top_k"),
                frequency_penalty=merged_params.get("frequency_penalty"),
                presence_penalty=merged_params.get("presence_penalty"),
                stop_sequences=merged_params.get("stop_sequences"),
                seed=merged_params.get("seed"),
                template_vars=self._extract_template_vars(),
            )
        else:
            llm_core = self.llm_core

        try:
            # Process through LLMCore
            # LLMCore.template_vars has agent's template vars; message.inputs adds runtime overrides
            llm_result = await llm_core.process_with_llm(
                template_vars=message.inputs or {},
                record=message.record,
                context=message.context,
                parent_trace_id=message.parent_call_id,
                cancellation_token=cancellation_token,
            )
            # Prepare metadata for AgentOutput
            output_metadata = {
                "agent_name": self.agent_name,
                "resolved_inputs": llm_result.resolved_inputs,
                "agent_id": self.agent_id,
                **llm_result.metadata,
            }

            logger.debug(f"Agent '{self.agent_name}' completed _process. Output type: {type(llm_result.content).__name__}")
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=llm_result.content,
                messages=llm_result.messages,
                metadata=output_metadata,
                error=[],
            )

        except ProcessingError as e:
            logger.error(f"Agent '{self.agent_id}': LLM processing failed: {e}")
            raise
