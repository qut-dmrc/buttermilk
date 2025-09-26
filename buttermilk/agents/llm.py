"""LLM-based Agent for Buttermilk.

This module provides the `LLMAgent` class, a foundational agent that integrates
with large language models (LLMs) to process tasks. It extends the base `Agent`
class with the ability to:
1. Fill prompt templates with input data and conversation context
2. Call LLMs with the rendered prompts
3. Parse structured outputs from LLM responses

`LLMAgent` serves as a base class for more specialized agents that require
LLM interaction. It now delegates core LLM operations to the shared `LLMCore`
class while maintaining Agent-specific functionality like message handling
and autogen integration.
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

        Now delegates core LLM functionality to LLMCore while maintaining
        Agent-specific behavior and autogen integration.

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

        # Store output model for reference
        self.output_model: type[pydantic.BaseModel] = output_model or None

        # Initialize the shared LLM core with our configuration
        self.llm_core = LLMCore(
            parameters=self.parameters,
            output_model=output_model,
            tools=self._tools  # Pass tools to LLMCore
        )

        # Keep model reference for compatibility
        self._model: str = self.parameters.get("model", "")

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput:
        """Core processing logic: uses LLMCore to process and creates AgentOutput.

        Now delegates the actual LLM operations to LLMCore while maintaining
        Agent-specific behavior like AgentInput/Output handling and context management.

        Args:
            message: The `AgentInput` message containing data, context, and
                parameters for this processing step.
            cancellation_token: Optional. A token to signal cancellation of the
                LLM call or other async operations.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput: An `AgentOutput` message with the processed LLM response
            and comprehensive metadata.

        Raises:
            ProcessingError: If LLM processing fails.
        """
        logger.debug(f"Agent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}.")

        try:
            # Merge parameters for template rendering
            task_params = message.parameters if message.parameters is not None else {}
            combined_params = {**self.parameters, **task_params}

            # Update LLMCore parameters with task-specific overrides
            self.llm_core.parameters = combined_params

            # Extract cancellation token from kwargs if not provided directly
            if cancellation_token is None:
                cancellation_token = kwargs.get("cancellation_token")

            # Call LLMCore with AgentInput data
            llm_result = await self.llm_core.process_with_llm(
                inputs=message.inputs if message.inputs is not None else {},
                context=message.context,
                records=message.records,
                parent_trace_id=message.parent_call_id,
                cancellation_token=cancellation_token,
            )

            logger.debug(f"Agent {self.agent_name}: Received result from LLMCore. Content type: {type(llm_result.content).__name__}")

            # Build comprehensive metadata for AgentOutput
            output_metadata = {
                "agent_name": self.agent_name,
                "agent_id": self.agent_id,
                "agent_model": self.parameters["model"],
                **llm_result.metadata,  # Include all LLM metadata (usage, pricing, etc)
                **llm_result.template_metadata,  # Include template metadata
            }

            # Store template metadata for ExecutionTrace compatibility
            self._template_metadata = llm_result.template_metadata

            # Extract the final output
            final_output = llm_result.content

            logger.debug(f"Agent '{self.agent_name}' completed _process. Output type: {type(final_output).__name__}")
            return AgentOutput(agent_id=self.agent_id, outputs=final_output, metadata=output_metadata, error=None)

        except ProcessingError as e:
            # Re-raise ProcessingError as-is
            logger.error(f"Agent '{self.agent_id}': Processing error: {e}")
            raise

        except Exception as e:
            # Wrap unexpected errors
            msg = f"Unexpected error in agent '{self.agent_id}': {e}"
            logger.error(msg, exc_info=True)
            raise ProcessingError(msg) from e
