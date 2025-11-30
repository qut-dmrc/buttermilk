"""Classification API processors for Buttermilk.

This module provides stateless classification processors that call external
classification APIs (Zentropi, Perspective, HuggingFace, etc.) and return
structured labels/scores.

Unlike LLMAgent which uses LLMs for reasoning, classifiers call external
APIs that return pre-defined categories and confidence scores. They follow
the same design pattern as LLMCore: stateless processors with tracing.
"""

import time
import uuid
from abc import abstractmethod
from typing import Any, AsyncGenerator

import pydantic
from opentelemetry import trace
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template
from buttermilk.utils.validators import import_class_from_path


class ClassifierResult(BaseModel):
    """Lightweight result structure for classifier operations.

    Mirrors LLMResult for consistency across processor types.
    """

    content: Any = Field(..., description="The classification output - structured object")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="API response, timing, model info"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for correlation",
    )
    template_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Template name, hash, etc"
    )
    error: str | None = Field(None, description="Error message if processing failed")


class ClassifierCore:
    """Stateless classification processor with template support.

    Design mirrors LLMCore: stateless, traceable, works directly with BaseRecord.
    No Agent inheritance, no AgentInput/AgentOutput - just process() and yield.

    Workflow:
    1. Render template with record data to create text
    2. Call _classify() with rendered text (subclass implements API call)
    3. Map API response to output_model schema (subclass implements mapping)
    4. Yield enriched BaseRecord with classification results

    Example:
        ```python
        class HateSpeechClassification(BaseModel):
            label: int
            confidence: float
            categories: list[str]

        classifier = ZentropiClassifier(
            template="classification_prompt",
            output_model=HateSpeechClassification,
        )

        async for record in classifier.process(input_record, processor_stage="classify"):
            print(record.output)
        ```
    """

    def __init__(
        self,
        *,
        template: str,
        output_model: type[pydantic.BaseModel] | str,
        output_col: str = "output",
        **kwargs: Any,
    ) -> None:
        """Initialize classifier with configuration.

        Args:
            template: Name of the prompt template to use.
            output_model: Pydantic model class or string path for structured output.
            output_col: Name of the output column in the record (default: "output").
            **kwargs: Additional configuration (stored in self.parameters).

        Raises:
            ValueError: If template or output_model is not specified or cannot be resolved.
        """
        if not template:
            raise ValueError("'template' is required for ClassifierCore")
        if not output_model:
            raise ValueError("'output_model' is required for ClassifierCore")

        self.template = template

        # Resolve output_model if it's a string path
        if isinstance(output_model, str):
            try:
                self.output_model = import_class_from_path(output_model)
            except Exception as e:
                raise ValueError(f"Failed to resolve output_model '{output_model}': {e}") from e
        else:
            self.output_model = output_model

        self.output_col = output_col
        self.parameters = kwargs

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord through classification.

        Args:
            record: Input BaseRecord to classify
            processor_stage: Unique stage identifier for tracing
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments passed to template rendering

        Yields:
            BaseRecord: Enriched record with classification output and metadata
        """
        tracer = trace.get_tracer("buttermilk.classifier")
        start_time = time.time()
        record_id = getattr(record, "record_id", "unknown")

        span_attributes = {
            "classifier.class": self.__class__.__name__,
            "classifier.template": self.template,
            "processor.stage": processor_stage,
            "record.id": record_id,
        }

        with tracer.start_as_current_span(
            f"classifier.{self.__class__.__name__}", attributes=span_attributes
        ) as span:
            try:
                result = await self._classify_record(record, **kwargs)

                # Build processing metadata
                processing_time_ms = int((time.time() - start_time) * 1000)
                stage_metadata = {
                    "classifier": self.__class__.__name__,
                    "template": self.template,
                    "template_hash": result.template_metadata.get("hash"),
                    "trace_id": result.trace_id,
                    "processing_time_ms": processing_time_ms,
                    "api_response": result.metadata.get("api_response"),
                }

                # Extract structured output
                output_content = result.content
                if hasattr(output_content, "model_dump"):
                    output_content = output_content.model_dump()

                # Enrich record
                enriched_record = record.model_copy(
                    update={
                        self.output_col: output_content,
                        "metadata": {
                            **(record.metadata or {}),
                            processor_stage: stage_metadata,
                        },
                    }
                )

                span.set_attribute("processing.time_ms", processing_time_ms)
                span.set_status(trace.Status(trace.StatusCode.OK))

                yield enriched_record

            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.error(
                    f"Classification failed for record {record_id}: {e}",
                    record_id=record_id,
                    processor_stage=processor_stage,
                    error=str(e),
                )
                raise ProcessingError(f"Classification failed: {e}") from e

    async def _classify_record(
        self, record: BaseRecord, **kwargs: Any
    ) -> ClassifierResult:
        """Internal method to classify a record.

        Args:
            record: Input record to classify
            **kwargs: Additional template variables

        Returns:
            ClassifierResult with structured output and metadata
        """
        # Step 1: Render template
        inputs = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        inputs.update(kwargs)

        try:
            rendered_text, unfilled_vars, template_hash = load_template(
                template=self.template,
                parameters=self.parameters,
                untrusted_inputs=inputs,
            )
            logger.debug(
                f"Classifier rendered template '{self.template}', "
                f"unfilled vars: {unfilled_vars}, hash: {template_hash}"
            )
        except Exception as e:
            raise ProcessingError(f"Template rendering failed: {e}") from e

        # Step 2: Call classification API
        try:
            api_response = await self._classify(rendered_text)
            logger.debug(f"Classifier received API response: {type(api_response).__name__}")
        except Exception as e:
            raise ProcessingError(f"Classification API call failed: {e}") from e

        # Step 3: Map to schema
        try:
            structured_output = self._map_to_schema(api_response, self.output_model)
            logger.debug(f"Classifier mapped to schema: {type(structured_output).__name__}")
        except Exception as e:
            raise ProcessingError(
                f"Failed to map response to {self.output_model.__name__}: {e}"
            ) from e

        return ClassifierResult(
            content=structured_output,
            metadata={"api_response": api_response},
            template_metadata={
                "name": self.template,
                "hash": template_hash,
                "unfilled_vars": list(unfilled_vars),
            },
        )

    @abstractmethod
    async def _classify(self, text: str) -> dict[str, Any]:
        """Call classification API with rendered text.

        Subclasses MUST implement this to perform the API call.

        Args:
            text: Rendered template text to classify

        Returns:
            Raw API response as dictionary

        Raises:
            Exception: If API call fails
        """
        raise NotImplementedError("Subclasses must implement _classify()")

    @abstractmethod
    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map API response to Pydantic output schema.

        Subclasses MUST implement this to convert API response to schema.

        Args:
            response: Raw API response from _classify()
            schema: Target Pydantic model class

        Returns:
            Instance of schema with mapped values

        Raises:
            ValueError: If response cannot be mapped
            pydantic.ValidationError: If validation fails
        """
        raise NotImplementedError("Subclasses must implement _map_to_schema()")


class HuggingFaceClassifier(ClassifierCore):
    """HuggingFace text classification model with template support.

    Uses LiteLLM to call HuggingFace inference endpoints for classification.

    Example:
        ```python
        class SafeguardClassification(BaseModel):
            label: int
            confidence: float
            category: str

        classifier = HuggingFaceClassifier(
            template="classification_prompt",
            model="gpt-oss-safeguard-20b",
            output_model=SafeguardClassification,
        )
        ```
    """

    def __init__(self, *, model: str, **kwargs: Any) -> None:
        """Initialize HuggingFaceClassifier.

        Args:
            model: Name of the HuggingFace model to use.
            **kwargs: Passed to ClassifierCore.

        Raises:
            ValueError: If model is not found in LLM connections.
        """
        super().__init__(**kwargs)

        if not model:
            raise ValueError("'model' is required for HuggingFaceClassifier")

        self.model = model

        # Get LLM wrapper from buttermilk connections
        from buttermilk import bm

        try:
            self._llm_wrapper = bm.llms.get_autogen_chat_client(self.model)
            logger.debug(f"HuggingFaceClassifier initialized with model: {self.model}")
        except AttributeError as e:
            raise ValueError(
                f"Model '{self.model}' not found in LLM connections. "
                f"Available: {list(bm.llms.connections.keys())}"
            ) from e

    async def _classify(self, text: str) -> dict[str, Any]:
        """Call HuggingFace classification model via LiteLLM."""
        from autogen_core.models import UserMessage

        try:
            messages = [UserMessage(content=text, source="user")]
            result = await self._llm_wrapper.create(messages=messages, schema=None)

            if hasattr(result, "content") and result.content:
                import json

                if isinstance(result.content, str):
                    response = json.loads(result.content)
                else:
                    response = result.content

                logger.debug(f"HuggingFace response: {response}")
                return response
            else:
                raise ProcessingError(f"Empty response from HuggingFace: {result}")

        except Exception as e:
            logger.error(f"HuggingFace classification failed: {e}")
            raise ProcessingError(f"HuggingFace classification failed: {e}") from e

    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map HuggingFace classification response to output schema."""
        schema_fields = schema.model_fields.keys()

        # Handle list response (multiple predictions)
        if isinstance(response, list) and len(response) > 0:
            prediction = response[0]
        else:
            prediction = response

        # Extract and convert label
        raw_label = prediction.get("label", "")
        if isinstance(raw_label, str):
            if raw_label.startswith("LABEL_"):
                try:
                    label = int(raw_label.split("_")[1])
                except (IndexError, ValueError):
                    label = 1 if raw_label != "LABEL_0" else 0
            else:
                label = 1 if raw_label.lower() in {"unsafe", "toxic", "hate", "1", "true"} else 0
        else:
            label = int(raw_label)

        confidence = float(prediction.get("score", 0.0))
        category = str(prediction.get("label", "unknown"))

        available_mapping = {
            "label": label,
            "confidence": confidence,
            "category": category,
            "categories": [category] if category else [],
        }

        # Only include fields schema accepts
        filtered_mapping = {k: v for k, v in available_mapping.items() if k in schema_fields}

        try:
            return schema(**filtered_mapping)
        except pydantic.ValidationError as e:
            logger.error(f"Failed to map HuggingFace response to {schema.__name__}: {e}")
            raise


class ZentropiClassifier(ClassifierCore):
    """Zentropi API classifier with template support.

    Zentropi provides toxicity classification returning:
    - label (str): Classification label (e.g., "1" for toxic, "0" for safe)
    - confidence (float): Confidence score
    - compute_time (float): Time taken for classification

    The API requires:
    - content_text: The text to be classified
    - criteria_text: The classification criteria (loaded from template)

    Example:
        ```python
        class ToxicityClassification(BaseModel):
            label: int
            confidence: float
            categories: list[str]

        classifier = ZentropiClassifier(
            template="toxicity_prompt",
            output_model=ToxicityClassification,
        )
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ZentropiClassifier with API credentials.

        Raises:
            ValueError: If ZENTROPI_API_KEY is not in environment.
        """
        super().__init__(**kwargs)

        import os

        api_key = os.environ.get("ZENTROPI_API_KEY")
        if not api_key:
            raise ValueError("ZENTROPI_API_KEY environment variable is required")

        base_url = os.environ.get("ZENTROPI_BASE_URL", "https://api.zentropi.ai/v1/label")

        self._client = {"api_key": api_key, "base_url": base_url}
        logger.debug(f"ZentropiClassifier initialized with base_url: {base_url}")

    async def _classify(self, text: str, *, content: str) -> dict[str, Any]:
        """Call Zentropi API with criteria and content.

        Args:
            text: Rendered template containing classification criteria
            content: Text content to classify

        Returns:
            Dictionary with 'label', 'confidence', and 'compute_time'

        Raises:
            ValueError: If response is missing 'label' field
            requests.exceptions.RequestException: If API call fails
        """
        import requests

        try:
            payload = {
                "content_text": content,
                "criteria_text": text,
            }

            response = requests.post(
                self._client["base_url"],
                headers={"Authorization": f"Bearer {self._client['api_key']}"},
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if "label" not in result:
                raise ValueError(f"Zentropi response missing 'label' field: {result.keys()}")

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Zentropi API call failed: {e}")
            raise

    async def _classify_record(
        self, record: BaseRecord, **kwargs: Any
    ) -> ClassifierResult:
        """Override to extract content separately for Zentropi API.

        Zentropi requires two separate fields:
        - criteria_text: Classification instructions (from template)
        - content_text: Text to classify (from record.content)

        Args:
            record: Input record to classify
            **kwargs: Additional template variables

        Returns:
            ClassifierResult with structured output and metadata

        Raises:
            ProcessingError: If template rendering, API call, or mapping fails
            ValueError: If record.content is None or empty
        """
        from buttermilk._core.exceptions import ProcessingError
        from buttermilk.utils.templating import load_template

        # Extract content from record
        if not record.content:
            raise ValueError("record.content cannot be None or empty for classification")

        content = str(record.content)

        # Step 1: Render template (this becomes criteria_text)
        inputs = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        inputs.update(kwargs)

        try:
            rendered_text, unfilled_vars, template_hash = load_template(
                template=self.template,
                parameters=self.parameters,
                untrusted_inputs=inputs,
            )
            logger.debug(
                f"ZentropiClassifier rendered template '{self.template}', "
                f"unfilled vars: {unfilled_vars}, hash: {template_hash}"
            )
        except Exception as e:
            raise ProcessingError(f"Template rendering failed: {e}") from e

        # Step 2: Call Zentropi API with both criteria and content
        try:
            api_response = await self._classify(rendered_text, content=content)
            logger.debug(f"ZentropiClassifier received API response: {type(api_response).__name__}")
        except Exception as e:
            raise ProcessingError(f"Zentropi API call failed: {e}") from e

        # Step 3: Map to schema
        try:
            structured_output = self._map_to_schema(api_response, self.output_model)
            logger.debug(f"ZentropiClassifier mapped to schema: {type(structured_output).__name__}")
        except Exception as e:
            raise ProcessingError(
                f"Failed to map response to {self.output_model.__name__}: {e}"
            ) from e

        return ClassifierResult(
            content=structured_output,
            metadata={"api_response": api_response},
            template_metadata={
                "name": self.template,
                "hash": template_hash,
                "unfilled_vars": list(unfilled_vars),
            },
        )

    def _map_to_schema(
        self, response: dict[str, Any], schema: type[pydantic.BaseModel]
    ) -> pydantic.BaseModel:
        """Map Zentropi API response to output schema.

        Zentropi returns:
        - label: str (e.g., "1" for toxic, "0" for safe)
        - confidence: float
        - compute_time: float
        """
        schema_fields = schema.model_fields.keys()

        # Convert string label to int: "1" → 1 (toxic), "0" → 0 (safe)
        label_str = response.get("label", "0")
        label = int(label_str)

        confidence = response.get("confidence", 0.0)

        # Zentropi doesn't return categories, set empty list as fallback
        categories: list[str] = []

        available_mapping = {
            "label": label,
            "confidence": confidence,
            "categories": categories,
        }

        filtered_mapping = {k: v for k, v in available_mapping.items() if k in schema_fields}

        try:
            return schema(**filtered_mapping)
        except pydantic.ValidationError as e:
            logger.error(f"Failed to map Zentropi response to {schema.__name__}: {e}")
            raise
