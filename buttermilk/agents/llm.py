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

    def __init__(
        self, *, output_model: type[pydantic.BaseModel] = None, **kwargs: Any
    ) -> None:
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
            raise ValueError(
                f"Agent {self.agent_name}: 'model' is required in agent parameters."
            )
        if "template" not in self.parameters:
            raise ValueError(
                f"Agent {self.agent_name}: 'template' is required in agent parameters."
            )

        # Initialize private attributes
        self.output_model: type[pydantic.BaseModel] = output_model or None

        # Initialize the shared LLM core
        # Filter out parameters that we're passing explicitly to avoid duplicates
        filtered_params = {
            k: v
            for k, v in self.parameters.items()
            if k not in ("output_model", "tools", "fail_on_unfilled_parameters")
        }
        self.llm_core = LLMCore(
            output_model=output_model,
            tools=self._tools or [],
            fail_on_unfilled_parameters=self.parameters.get(
                "fail_on_unfilled_parameters", True
            ),
            **filtered_params,
        )

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
        logger.debug(
            f"Agent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}."
        )

        # Pass the entire message object to LLMCore for flexible input handling
        # LLMCore will extract inputs, context, and record as needed

        # Override parameters with message parameters
        if message.parameters:
            # Create a new LLMCore instance with runtime parameters
            # Template is set at init time and should not be overridden
            merged_params = {**self.parameters, **message.parameters}
            # Filter out parameters we're passing explicitly to avoid duplicates
            # Template comes from agent config only, not message parameters
            filtered_params = {
                k: v
                for k, v in merged_params.items()
                if k
                not in (
                    "model",
                    "output_model",
                    "tools",
                    "fail_on_unfilled_parameters",
                    "template",
                )
            }
            llm_core = LLMCore(
                model=merged_params.get("model", ""),
                template=self.parameters.get(
                    "template", ""
                ),  # Use agent's template, not message override
                output_model=self.output_model,
                tools=self._tools or [],
                fail_on_unfilled_parameters=merged_params.get(
                    "fail_on_unfilled_parameters", True
                ),
                **filtered_params,
            )
        else:
            llm_core = self.llm_core

        try:
            # Process through LLMCore (yields LLMResult)
            llm_result = await llm_core.process_with_llm(
                template_vars=message.inputs,
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

            logger.debug(
                f"Agent '{self.agent_name}' completed _process. Output type: {type(llm_result.content).__name__}"
            )
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
