import asyncio
from typing import Any, Optional

from pydantic import Field, PrivateAttr
import weave

from buttermilk._core.agent import Agent, AgentConfig, AgentInput, AgentOutput, FatalError, ProcessingError
from buttermilk._core.contract import StepRequest, ToolOutput, OOBMessages  # Added ToolOutput, OOBMessages
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import Record, RunRequest
from buttermilk.bm import logger


class BatchOrchestrator(Orchestrator):
    """
    Orchestrator for running flows in batch mode without user interaction.

    Executes steps based on a predefined sequence or simple logic,
    handling multiple variants according to configuration. Requires the
    `_run` method to be fully implemented with step sequencing and variant handling logic.
    """

    # Example configuration option for execution strategy
    execution_strategy: str = Field(
        default="all_combinations",
        description="How to handle agent variants ('all_combinations', 'group_by_criterion', etc.)",
    )
    _agent_instances: dict[str, Agent] = PrivateAttr(default_factory=dict)  # Store agent instances by ID

    async def _setup(self):
        """
        Set up resources for batch execution.

        This involves instantiating all agent variants defined in the configuration
        and storing them for later execution.
        """
        logger.info(f"Setting up BatchOrchestrator {self.session_id} for flow '{self.name}'...")

        # Instantiate agents based on configuration
        for step_name, variants_config in self.agents.items():
            # variants_config is AgentVariants, call get_configs()
            for agent_cls, variant_config in variants_config.get_configs():
                try:
                    # Assuming agent_cls is the actual class type for now
                    # If agent_cls is a string path, use Hydra/OmegaConf instantiate
                    instance = agent_cls(**variant_config.model_dump())
                    # Initialize the agent (e.g., load resources)
                    await instance.initialize()
                    self._agent_instances[variant_config.id] = instance  # Store by unique variant ID
                    logger.debug(f"Instantiated agent variant: {variant_config.id} ({step_name})")
                except Exception as e:
                    logger.error(f"Failed to instantiate agent variant {variant_config.id}: {e}", exc_info=True)
                    # Optionally raise FatalError to halt if agent is critical

        logger.info("BatchOrchestrator setup complete.")

    async def _cleanup(self):
        """Clean up any resources used by the batch orchestrator."""
        # TODO: Add specific cleanup if needed (e.g., closing agent resources)
        logger.info(f"Cleaning up BatchOrchestrator {self.session_id}...")
        # Placeholder: Close connections, release resources.
        await asyncio.sleep(0.1)  # Simulate cleanup
        logger.info("BatchOrchestrator cleanup complete.")

    async def _execute_step(
        self,
        step: StepRequest,
    ) -> AgentOutput | None:
        """
        Executes a single step of the flow for the given agent role.

        Handles variant selection based on internal logic (currently basic: uses first variant).

        Args:
            step: StepRequest with details about the step to run.

        Returns:
            The AgentOutput from the agent, or None if execution failed,
            or an AgentOutput with error info if execution raised an exception.
        """
        # --- Variant Selection Logic (Placeholder: Use first variant) ---
        variants_config = self.agents.get(step.role.lower())
        if not variants_config or not variants_config.variants:
            logger.error(f"No variants found for step '{step.role}'. Skipping.")
            return AgentOutput(agent_id=self.name, error=[f"No variants configured for step: {step.role}"])

        # Get the config for the first variant
        first_variant_config = variants_config.variants[0]
        logger.info(f"Executing step '{step.role}' (using variant {first_variant_config.id})...")
        agent = self._get_agent_instance(first_variant_config.id)
        # --- End Variant Selection ---

        if agent:
            try:
                message = AgentInput(prompt=step.prompt)
                # Assuming agent is callable via __call__ which calls _process
                # Agent._process is already traced by weave
                raw_output = await agent(message=message, **{})  # Pass empty kwargs for now

                # Handle different output types
                if isinstance(raw_output, AgentOutput):
                    output = raw_output  # Assign to output if it's the expected type
                    # if not output.is_error:
                    #     # Call evaluation if possible
                    #     ground_truth_record = next((r for r in input.records if getattr(r, "ground_truth", None) is not None), None)
                    #     criteria = input.parameters.get("criteria") or self.params.get("criteria")
                    #     # TODO: Implement _evaluate_step for BatchOrchestrator
                    #     # await self._evaluate_step(output, ground_truth_record, criteria, None) # Pass None for weave_call
                    #     logger.warning("_evaluate_step not implemented for BatchOrchestrator yet.")
                    logger.info(f"Step '{step.role}' variant '{first_variant_config.id}' completed.")
                    return output  # Return AgentOutput
                elif isinstance(raw_output, (ToolOutput, OOBMessages)):
                    logger.warning(
                        f"Step '{step.role}' variant '{first_variant_config.id}' returned unexpected type {type(raw_output)} in batch mode. Ignoring."
                    )
                    return None  # Don't propagate ToolOutput/OOB in simple batch mode
                elif raw_output is None:
                    logger.warning(f"Step '{step.role}' variant '{first_variant_config.id}' returned None.")
                    return None
                else:
                    logger.error(f"Step '{step.role}' variant '{first_variant_config.id}' returned unknown type {type(raw_output)}.")
                    return AgentOutput(agent_id=self.name, error=[f"Unknown return type: {type(raw_output)}"])

            except Exception as e:
                logger.error(f"Error executing step '{step.role}' variant '{first_variant_config.id}': {e}", exc_info=True)
                # Ensure 'inputs' attribute exists on output for error cases
                return AgentOutput(agent_id=self.name, error=[str(e)])
        else:
            # Error already logged by _get_agent_instance
            # Ensure 'inputs' attribute exists on output for error cases
            return AgentOutput(agent_id=self.name, error=[f"Agent instance not found: {first_variant_config.id}"])

    async def _run(self, request: RunRequest | None = None) -> None:
        """
        Main execution method for batch processing.

        This method should implement the core logic for determining the sequence
        of steps to run and how to handle the configured agent variants based on
        the `execution_strategy`. This placeholder needs to be fully implemented.
        """
        try:
            await self._setup()

            # TODO: Implement step sequencing logic based on flow definition or strategy
            logger.warning("BatchOrchestrator _run logic (step sequencing, variant handling) is not implemented.")

        except (StopAsyncIteration, KeyboardInterrupt) as e:
            logger.info(f"Batch run interrupted: {type(e).__name__}")
        except FatalError as e:
            logger.exception(f"Fatal error during batch run: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during batch run: {e}")
        finally:
            await self._cleanup()

    # Implemented helper method
    def _get_agent_instance(self, agent_id: str) -> Optional[Agent]:
        """
        Retrieve an instantiated agent instance by its unique configuration ID.

        Args:
            agent_id: The unique ID of the agent variant configuration.

        Returns:
            The instantiated Agent object or None if not found.
        """
        instance = self._agent_instances.get(agent_id)
        if not instance:
            logger.error(f"Agent instance with ID '{agent_id}' not found.")
        return instance
