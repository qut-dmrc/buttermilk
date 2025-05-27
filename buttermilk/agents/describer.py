"""Provides the Describer agent for generating textual descriptions of media content.

This module defines the `Describer` agent, which is a specialized `LLMAgent`
designed to analyze media objects (images, videos, audio) within a `Record`
and generate a textual description (often an alt text or caption) using a
configured Language Model.
"""

from typing import Any  # For type hinting

from PIL.Image import Image  # For image type checking
from buttermilk import logger  # Buttermilk's centralized logger
from buttermilk._core.agent import AgentInput  # Buttermilk AgentInput type
from buttermilk._core.contract import AgentTrace, ErrorEvent  # Buttermilk contract types
from buttermilk.agents.llm import LLMAgent  # Base LLM Agent
from buttermilk.utils.media import download_and_convert  # Utility for media handling


class Describer(LLMAgent):
    """An agent that generates textual descriptions for media objects using an LLM.

    The `Describer` agent extends `LLMAgent` to specifically focus on tasks
    like creating alt text for images, transcribing audio, or summarizing video
    content. It checks if a description (`alt_text`) already exists or if the
    content is purely textual before invoking the LLM. It can also download
    media from a URI if necessary.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required (inherited from LLMAgent)**. The name of the
          LLM to use for generating descriptions.
        - `prompt_template` (str): **Required (inherited from LLMAgent)**. The name
          of the prompt template to use. For `Describer`, this template should
          guide the LLM to describe the provided media content. Defaults to "describe".
        - `download_if_necessary` (bool): Optional. If `True` (default), the agent
          will attempt to download media from a URI specified in the input if
          the `Record` does not already contain the media components.

    Input:
        Expects an `AgentInput` message. The relevant information is typically
        within `message.records`, where each `Record` might contain media
        components (images, video, audio) or a URI to such media. If `message.records`
        is empty, it might use `message.inputs` (if `download_if_necessary` is True)
        to fetch a record.

    Output:
        Produces an `AgentTrace` containing the LLM's generated description in its
        `outputs` field. This description is also used to update the `alt_text`
        attribute of the processed `Record`. If processing is skipped or fails,
        it might return the original job data or an `ErrorEvent`.

    Attributes:
        template (str | None): The default prompt template name to use if not
            otherwise specified in parameters. Defaults to "describe".
            (Note: This seems to be a class attribute, but `LLMAgent` typically
            expects `prompt_template` in `parameters`).
    """
    template: str | None = "describe" # Default prompt template for this agent

    async def _process( # Overriding _process from base Agent/LLMAgent
        self,
        *,
        message: AgentInput, # Changed 'job' to 'message' to align with Agent._process
        **kwargs: Any,
    ) -> AgentTrace | ErrorEvent | None: # Return type consistent with LLMAgent._process
        """Processes the input message to generate a description for its media content.

        This method orchestrates the description generation by:
        1.  Optionally downloading media if `message.records` is empty but URIs are
            provided in `message.inputs` and `download_if_necessary` is true.
        2.  Checking if the record contains any non-text media components (image,
            video, audio). If not, or if no record/components are found, it skips LLM processing.
        3.  Checking if the record already has `alt_text`. If so, it skips LLM processing.
        4.  If checks pass, it calls the `super()._process()` (from `LLMAgent`)
            to invoke the LLM with the appropriate prompt and media.
        5.  Updates the `alt_text` of the input `Record` (and potentially other
            metadata via `record.update_from`) with the generated description from
            the LLM's output.

        Args:
            message: The `AgentInput` containing data and parameters for the description task.
                     The `message.records` (or `message.inputs` for download) should
                     provide the media to be described.
            **kwargs: Additional keyword arguments passed down from the caller.

        Returns:
            AgentTrace | ErrorEvent | None:
            - An `AgentTrace` containing the LLM's response and updated record information
              if description generation was successful.
            - An `ErrorEvent` if a significant error occurred during processing.
            - `None` if processing was intentionally skipped (e.g., no media,
              alt_text already exists). The original `message` (or its relevant parts)
              might be passed through by the orchestrator in such cases, or this
              `None` return can signify no action was taken by this agent.
              (Note: The original `process_job` returned the `job` object itself
              when skipping. `_process` in `Agent` expects `AgentOutput` or similar.
              Returning `None` signals no direct output from this agent for this input,
              which might be appropriate for a skip.)
        """
        # Ensure there's a record to process, either from message.records or by downloading
        record_to_process = None
        if message.records:
            record_to_process = message.records[0] # Assuming one record per AgentInput for describer
            # TODO: How should Describer handle multiple records in a single AgentInput?
            # For now, processing only the first one.
            logger.debug(f"Describer '{self.agent_id}' using record '{record_to_process.record_id}' from input.")
        elif self.parameters.get("download_if_necessary", True) and message.inputs:
            logger.debug(
                f"Describer '{self.agent_id}': No records in input, attempting to download from inputs: {list(message.inputs.keys())}.",
            )
            try:
                # download_and_convert expects kwargs that match its signature based on input type
                # This assumes message.inputs is structured appropriately
                record_to_process = await download_and_convert(**message.inputs)
                if record_to_process:
                     # If downloaded, add it to the message.records for consistency downstream
                     # and so that super()._process() can find it.
                    message.records = [record_to_process]
                logger.debug(f"Describer '{self.agent_id}' downloaded record: {record_to_process.record_id if record_to_process else 'Failed'}")
            except Exception as e:
                logger.error(f"Describer '{self.agent_id}': Failed to download record: {e!s}")
                return ErrorEvent(source=self.agent_id, content=f"Failed to download record: {e!s}")

        if not record_to_process:
            logger.debug(
                f"Describer '{self.agent_id}': No record available to process for message_id '{message.message_id if hasattr(message, 'message_id') else 'N/A'}'. Skipping.",
            )
            return None # No record, nothing to do

        # Check for non-text media components
        # The original code checked record.components which might not be populated if MediaObj is in record.content
        # A more robust check might involve iterating through record.content if it's a list of MediaObj
        media_exists = False
        if isinstance(record_to_process.content, Sequence) and not isinstance(record_to_process.content, str):
            for item in record_to_process.content:
                if isinstance(item, Image):
                    media_exists = True
                    break
        # Add other checks if record.components was the intended place for complex media
        # For now, assuming content holds MediaObj items for multimodal.

        if not media_exists:
            logger.debug(
                f"Describer '{self.agent_id}' for record '{record_to_process.record_id}': No non-text media components found. Skipping description.",
            )
            return None # No relevant media to describe

        # Skip if alt_text already exists
        if record_to_process.alt_text:
            logger.debug(
                f"Describer '{self.agent_id}' for record '{record_to_process.record_id}': Alt text already exists. Skipping.",
            )
            return None # Alt text already present

        # Call LLMAgent's _process method to get the description
        # The LLMAgent._process expects the record to be in message.records
        # and uses the prompt template (e.g., "describe")
        agent_trace_result = await super()._process(message=message, **kwargs)

        if isinstance(agent_trace_result, AgentTrace) and agent_trace_result.outputs:
            # Update record's alt_text and potentially other metadata from the LLM's output.
            # The structure of agent_trace_result.outputs depends on the LLM and prompt.
            # Assuming outputs is a dict or can be converted to one for update_from.
            output_data = agent_trace_result.outputs
            if isinstance(output_data, str): # If LLM output is just a string
                record_to_process.alt_text = output_data
                # Optionally, put this string into the AgentTrace.outputs as a structured dict if preferred
                agent_trace_result.outputs = {"generated_alt_text": output_data}
            elif isinstance(output_data, dict):
                # If output is a dict, try to find a key for alt_text or use a default key
                # This part depends on how the prompt template structures the LLM output.
                generated_description = output_data.get("description", output_data.get("alt_text", output_data.get("text")))
                if generated_description and isinstance(generated_description, str):
                    record_to_process.alt_text = generated_description

                # update_from expects a Pydantic model or dict.
                # It might be too broad here if outputs contains many things.
                # Consider more targeted updates to record_to_process.metadata if needed.
                try:
                    record_to_process.metadata.update(output_data) # Example: merge all outputs into metadata
                    # A more specific update would be:
                    # record_to_process.update_from_dict(output_data) # if Record has such a method
                except Exception as e:
                    logger.warning(f"Describer '{self.agent_id}': Failed to update record metadata from LLM outputs: {e!s}")
            else:
                logger.warning(f"Describer '{self.agent_id}': LLM output was not a string or dict, cannot directly update alt_text. Output: {output_data}")

            logger.info(f"Describer '{self.agent_id}' generated alt_text for record '{record_to_process.record_id}'.")
            # The agent_trace_result already contains the outputs from the LLM.
            # If record_to_process was part of message.records, it's a mutable object,
            # so modifications here will be reflected in the message object if it's passed around.
            return agent_trace_result
        elif isinstance(agent_trace_result, ErrorEvent):
            logger.error(f"Describer '{self.agent_id}': LLM processing failed for record '{record_to_process.record_id}': {agent_trace_result.content}")
            return agent_trace_result # Propagate the error
        else:
            logger.warning(f"Describer '{self.agent_id}': LLM processing for record '{record_to_process.record_id}' did not return a valid AgentTrace or ErrorEvent.")
            return None
