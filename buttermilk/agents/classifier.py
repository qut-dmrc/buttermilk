"""Classification API agents for Buttermilk.

This module provides the `ClassifierAgent` base class and implementations for
classification APIs (Zentropi, Perspective, etc.) that return structured labels/scores.

Unlike `LLMAgent` which uses LLMs for reasoning, `ClassifierAgent` calls external
classification APIs that return pre-defined categories and confidence scores.

Template support allows consistent input formatting across the pipeline, though
classification results may not vary significantly by template (API limitation).
"""

from abc import abstractmethod
from typing import Any, AsyncGenerator

import pydantic
from autogen_core import CancellationToken

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template


class ClassifierAgent(Agent):
    """Base class for classification API agents that support templates.

    Unlike LLMAgent which uses LLMs for reasoning, ClassifierAgent calls
    external classification APIs (Zentropi, Perspective, etc.) that return
    structured labels/scores without generative capabilities.

    Template support allows consistent input formatting across the pipeline,
    though classification results may not vary by template (API limitation).

    The agent workflow:
    1. Render template with inputs/record/context to create text
    2. Call _classify() with rendered text (subclass implements API call)
    3. Map API response to output_model schema (subclass implements mapping)
    4. Return AgentOutput with structured results

    Configuration (from `AgentConfig`):
        template: Name of the prompt template to use (required in parameters)
        output_model: Pydantic model for structured output (required)
        fail_on_unfilled_parameters: Whether to fail if template variables are missing

    Attributes:
        output_model: Pydantic model class for response mapping

    Example:
        ```python
        # Define output schema
        class HateSpeechClassification(BaseModel):
            label: int  # 0 = not hate speech, 1 = hate speech
            confidence: float
            categories: list[str]

        # Create classifier agent
        agent = ZentropiClassifier(
            agent_name="HateSpeechClassifier",
            role="HATE_SPEECH_CLASSIFIER",
            parameters={"template": "classification_prompt"},
            output_model=HateSpeechClassification,
        )

        # Invoke with input
        result = await agent.invoke(
            AgentInput(inputs={"text": "Example text to classify"})
        )
        ```
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        output_model: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a ClassifierAgent with the provided configuration.

        Args:
            template: Name of the prompt template to use.
            output_model: Pydantic model class for structured output mapping.
                If not provided, raises ValueError (fail-fast).
            **kwargs: Configuration parameters passed to AgentConfig.

        Raises:
            ValueError: If 'template' or 'output_model' is not specified.
        """
        super().__init__(**kwargs)

        # Template can come from direct arg or parameters dict
        self.template = template or self.parameters.get("template")

        # Fail-fast: require template
        if not self.template:
            raise ValueError(
                f"Agent {self.agent_name}: 'template' is required."
            )

        # Fail-fast: require output_model
        if output_model is None:
            raise ValueError(
                f"Agent {self.agent_name}: 'output_model' is required for ClassifierAgent."
            )

        self.output_model: type[pydantic.BaseModel] = output_model

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Processor interface for use in pipelines (e.g., VariantProcessor).

        Wraps the agent workflow with the standard Processor interface, allowing
        ClassifierAgent subclasses to be used in VariantProcessor configurations.

        Args:
            record: Input BaseRecord to classify
            processor_stage: Unique stage identifier for tracing
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments passed to template rendering

        Yields:
            BaseRecord: Enriched record with classification output and metadata
        """
        # Convert BaseRecord to AgentInput format
        inputs = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        inputs.update(kwargs)

        agent_input = AgentInput(inputs=inputs, record=record)

        # Call the agent's _process method directly
        agent_output = await self._process(
            message=agent_input,
            cancellation_token=None,
        )

        # Extract structured output
        output_content = agent_output.outputs
        if hasattr(output_content, "model_dump"):
            output_content = output_content.model_dump()

        # Enrich record with classification results
        enriched_record = record.model_copy(
            update={
                "output": output_content,
                "metadata": {
                    **record.metadata,
                    processor_stage: agent_output.metadata,
                },
            }
        )

        yield enriched_record

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AgentOutput:
        """Core processing logic: render template, classify, map to schema.

        Workflow:
        1. Render Jinja2 template with inputs/record/context
        2. Call _classify() with rendered text (API call)
        3. Map API response to output_model schema
        4. Return AgentOutput with structured results

        Args:
            message: The `AgentInput` message containing data, context, and
                parameters for this processing step.
            cancellation_token: Optional. A token to signal cancellation.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput: An `AgentOutput` message with the classification results.

        Raises:
            ProcessingError: If template rendering, API call, or schema mapping fails.
        """
        logger.debug(
            f"Agent '{self.agent_name}' starting _process for message_id: {getattr(message, 'message_id', 'N/A')}."
        )

        # Step 1: Render template with inputs
        try:
            rendered_text, unfilled_vars, template_hash = load_template(
                template=self.template,
                parameters=self.parameters,
                untrusted_inputs=message.inputs or {},
            )
            logger.debug(
                f"Agent '{self.agent_name}' rendered template '{self.template}', "
                f"unfilled vars: {unfilled_vars}, hash: {template_hash}"
            )
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}': Template rendering failed: {e}")
            raise ProcessingError(
                f"Failed to render template '{self.template}': {e}"
            ) from e

        # Step 2: Call classification API with rendered text
        try:
            api_response = await self._classify(rendered_text)
            logger.debug(
                f"Agent '{self.agent_name}' received API response: {type(api_response).__name__}"
            )
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_name}': Classification API call failed: {e}"
            )
            raise ProcessingError(f"Classification API call failed: {e}") from e

        # Step 3: Map API response to output_model schema
        try:
            structured_output = self._map_to_schema(api_response, self.output_model)
            logger.debug(
                f"Agent '{self.agent_name}' mapped response to schema: {type(structured_output).__name__}"
            )
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}': Schema mapping failed: {e}")
            raise ProcessingError(
                f"Failed to map API response to schema {self.output_model.__name__}: {e}"
            ) from e

        # Step 4: Return AgentOutput with results
        output_metadata = {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "template": self.template,
            "template_hash": template_hash,
            "unfilled_vars": list(unfilled_vars),
            "api_response": api_response,  # Include raw response for debugging
        }

        logger.debug(
            f"Agent '{self.agent_name}' completed _process. Output type: {type(structured_output).__name__}"
        )

        return AgentOutput(
            agent_id=self.agent_id,
            outputs=structured_output,
            metadata=output_metadata,
            error=[],
        )

    @abstractmethod
    async def _classify(self, text: str) -> dict[str, Any]:
        """Call classification API with rendered text.

        Subclasses MUST implement this method to perform the actual API call
        to their classification service (Zentropi, Perspective, etc.).

        Args:
            text: Rendered template text to classify

        Returns:
            Raw API response as dictionary. Structure depends on the API
            (e.g., {"toxic": bool, "scores": {...}, "labels": [...]})

        Raises:
            Exception: If API call fails (network error, auth error, etc.)
        """
        raise NotImplementedError("Subclasses must implement _classify()")

    @abstractmethod
    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map API response to Pydantic output schema.

        Subclasses MUST implement this method to convert their API's response
        format into the target output_model schema.

        Args:
            response: Raw API response dictionary from _classify()
            schema: Target Pydantic model class (self.output_model)

        Returns:
            Instance of schema populated with mapped values from response

        Raises:
            ValueError: If response cannot be mapped to schema
            pydantic.ValidationError: If mapped values fail schema validation
        """
        raise NotImplementedError("Subclasses must implement _map_to_schema()")


class HuggingFaceClassifier(ClassifierAgent):
    """HuggingFace text classification model with template support.

    Uses LiteLLM to call HuggingFace inference endpoints for classification tasks.
    Designed for models like gpt-oss-safeguard that perform text classification.

    Unlike LLM-based classifiers that generate text, this classifier uses HuggingFace
    models that return structured classification outputs (labels, scores, etc.).

    Example:
        ```python
        class SafeguardClassification(BaseModel):
            label: int  # 0 = safe, 1 = unsafe
            confidence: float
            category: str

        classifier = HuggingFaceClassifier(
            agent_name="SafeguardClassifier",
            role="SAFETY_CLASSIFIER",
            parameters={"template": "classification_prompt", "model": "gpt-oss-safeguard-20b"},
            output_model=SafeguardClassification,
        )
        ```

    Configuration:
        Requires 'model' in parameters (e.g., "gpt-oss-safeguard-20b")
        Model must be configured in buttermilk LLM connections (models.json)
        HuggingFace API key must be configured for the model
    """

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        """Initialize HuggingFaceClassifier with model configuration.

        Args:
            model: Name of the HuggingFace model to use.
            **kwargs: Configuration parameters including:
                - output_model: Pydantic model for classification output

        Raises:
            ValueError: If 'model' is not specified
            ValueError: If model is not found in LLM connections
        """
        super().__init__(**kwargs)

        # Model can come from direct arg or parameters dict
        model_name = model or self.parameters.get("model")

        # Fail-fast: require model
        if not model_name:
            raise ValueError(
                f"Agent {self.agent_name}: 'model' is required."
            )

        self.model = model_name

        # Get LLM wrapper from buttermilk connections
        from buttermilk import bm

        try:
            self._llm_wrapper = bm.llms.get_autogen_chat_client(self.model)
            logger.debug(
                f"Agent '{self.agent_name}' initialized HuggingFace classifier with model: {self.model}"
            )
        except AttributeError as e:
            raise ValueError(
                f"Agent {self.agent_name}: Model '{self.model}' not found in LLM connections. "
                f"Available models: {list(bm.llms.connections.keys())}"
            ) from e

    async def _classify(self, text: str) -> dict[str, Any]:
        """Call HuggingFace classification model via LiteLLM.

        Args:
            text: Rendered template text to classify

        Returns:
            Classification response as dictionary with model-specific structure.
            For gpt-oss-safeguard: {"label": str, "score": float, ...}

        Raises:
            ProcessingError: If API call fails or response is invalid
        """
        from buttermilk._core.contract import UserMessage

        try:
            # Create message for classification
            messages = [UserMessage(content=text, source="user")]

            # Call LLM wrapper (uses LiteLLM for HuggingFace models)
            result = await self._llm_wrapper.create(
                messages=messages,
                schema=None,  # Let HuggingFace return raw classification output
            )

            # Extract content from result
            if hasattr(result, "content") and result.content:
                import json

                # Parse JSON response
                if isinstance(result.content, str):
                    response = json.loads(result.content)
                else:
                    response = result.content

                logger.debug(
                    f"Agent '{self.agent_name}' received HuggingFace response: {response}"
                )
                return response
            else:
                raise ProcessingError(
                    f"Empty or invalid response from HuggingFace model: {result}"
                )

        except Exception as e:
            logger.error(f"HuggingFace classification call failed: {e}")
            raise ProcessingError(
                f"HuggingFace classification failed: {e}"
            ) from e

    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map HuggingFace classification response to output schema.

        Maps common HuggingFace classification fields to schema fields:
        - label (str/int) -> label (int): Convert string labels to binary/multi-class
        - score (float) -> confidence (float): Classification confidence
        - class_label (str) -> category (str): Human-readable category name

        Only passes fields that the schema actually accepts to support
        strict schemas with extra="forbid".

        Args:
            response: HuggingFace classification response
            schema: Target Pydantic model class

        Returns:
            Instance of schema with mapped values

        Raises:
            ValueError: If response cannot be mapped to schema
            pydantic.ValidationError: If mapped values fail validation
        """
        # Get schema's expected fields
        schema_fields = schema.model_fields.keys()

        # Extract classification outputs from response
        # HuggingFace text-classification returns: [{"label": "LABEL_1", "score": 0.95}]
        # or single dict: {"label": "LABEL_1", "score": 0.95}

        # Handle list response (multiple predictions)
        if isinstance(response, list) and len(response) > 0:
            prediction = response[0]  # Use top prediction
        else:
            prediction = response

        # Extract label and convert to int if needed
        raw_label = prediction.get("label", "")
        if isinstance(raw_label, str):
            # Common pattern: "LABEL_0", "LABEL_1" -> 0, 1
            if raw_label.startswith("LABEL_"):
                try:
                    label = int(raw_label.split("_")[1])
                except (IndexError, ValueError):
                    # Fallback: use 1 for any non-zero label
                    label = 1 if raw_label != "LABEL_0" else 0
            else:
                # Binary classification: map to 0/1
                label = 1 if raw_label.lower() in {"unsafe", "toxic", "hate", "1", "true"} else 0
        else:
            label = int(raw_label)

        # Extract confidence score
        confidence = float(prediction.get("score", 0.0))

        # Extract category (if present)
        category = prediction.get("label", "unknown")

        # Build mapping dict with all available fields
        available_mapping = {
            "label": label,
            "confidence": confidence,
            "category": category,
            "categories": [category] if category else [],
        }

        # Only include fields that schema accepts (supports extra="forbid")
        filtered_mapping = {
            k: v for k, v in available_mapping.items() if k in schema_fields
        }

        # Attempt to construct schema instance
        try:
            return schema(**filtered_mapping)
        except pydantic.ValidationError as e:
            logger.error(
                f"Failed to map HuggingFace response to schema {schema.__name__}: {e}"
            )
            raise


class ZentropiClassifier(ClassifierAgent):
    """Zentropi API classifier with template support.

    Zentropi provides a toxicity classification API that returns:
    - toxic (bool): Whether content is toxic
    - scores (dict): Score values by measure name
    - labels (list): List of detected labels

    The classifier maps this to the provided output_model schema.

    Example output_model mapping:
        Input (Zentropi API):
            {"toxic": true, "scores": {"toxicity": 0.85}, "labels": ["profanity"]}

        Output (mapped to schema):
            HateSpeechClassification(label=1, confidence=0.85, categories=["profanity"])

    Configuration:
        Requires ZENTROPI_API_KEY in environment or credentials dict
        Optional ZENTROPI_BASE_URL (defaults to https://api.zentropi.ai/v1/label)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ZentropiClassifier with API credentials.

        Raises:
            ValueError: If ZENTROPI_API_KEY is not found in environment
        """
        super().__init__(**kwargs)

        # Initialize Zentropi client
        import os

        api_key = os.environ.get("ZENTROPI_API_KEY")
        if not api_key:
            raise ValueError(
                f"Agent {self.agent_name}: ZENTROPI_API_KEY environment variable is required"
            )

        base_url = os.environ.get(
            "ZENTROPI_BASE_URL", "https://api.zentropi.ai/v1/label"
        )

        self._client = {
            "api_key": api_key,
            "base_url": base_url,
        }

        logger.debug(
            f"Agent '{self.agent_name}' initialized Zentropi client with base_url: {base_url}"
        )

    async def _classify(self, text: str) -> dict[str, Any]:
        """Call Zentropi API with text.

        Args:
            text: Rendered template text to classify

        Returns:
            Zentropi API response:
                {
                    "toxic": bool,
                    "scores": {"toxicity": float, ...},
                    "labels": [str, ...]
                }

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If response is missing required fields
        """
        import requests

        try:
            response = requests.post(
                self._client["base_url"],
                headers={"Authorization": f"Bearer {self._client['api_key']}"},
                json={"text": text},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Validate required fields (fail-fast)
            if "toxic" not in result:
                raise ValueError(
                    f"Zentropi response missing required 'toxic' field. Got: {result.keys()}"
                )

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Zentropi API call failed: {e}")
            raise

    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map Zentropi API response to output schema.

        Maps Zentropi fields to common classification schema patterns:
        - toxic (bool) -> label (int): 1 if toxic, 0 if not
        - scores (dict) -> confidence (float): max score value
        - labels (list) -> categories (list): pass through

        Only passes fields that the schema actually accepts to support
        strict schemas with extra="forbid".

        Args:
            response: Zentropi API response with "toxic", "scores", "labels"
            schema: Target Pydantic model class

        Returns:
            Instance of schema with mapped values

        Raises:
            ValueError: If schema doesn't have expected fields
            pydantic.ValidationError: If mapped values fail validation
        """
        # Get schema's expected fields
        schema_fields = schema.model_fields.keys()

        # Map toxic bool to label int (common pattern: 1=toxic, 0=not toxic)
        label = 1 if response["toxic"] else 0

        # Extract confidence from scores (use max score value)
        confidence = 0.0
        if "scores" in response and response["scores"]:
            confidence = max(response["scores"].values())

        # Pass through labels as categories
        categories = response.get("labels", [])

        # Build mapping dict with all available fields
        available_mapping = {
            "label": label,
            "confidence": confidence,
            "categories": categories,
        }

        # Only include fields that schema accepts (supports extra="forbid")
        filtered_mapping = {
            k: v for k, v in available_mapping.items() if k in schema_fields
        }

        # Attempt to construct schema instance
        try:
            return schema(**filtered_mapping)
        except pydantic.ValidationError as e:
            logger.error(
                f"Failed to map Zentropi response to schema {schema.__name__}: {e}"
            )
            raise
