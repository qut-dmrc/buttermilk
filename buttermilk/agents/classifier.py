"""Classification API processors for Buttermilk.

This module provides stateless classification processors that call external
classification APIs (Zentropi, Perspective, HuggingFace, etc.) and return
structured labels/scores.

Unlike LLMAgent which uses LLMs for reasoning, classifiers call external
APIs that return pre-defined categories and confidence scores. They follow
the same design pattern as LLMCore: stateless processors with tracing.

ClassifierCore extends ProcessorCore to share common infrastructure
(trace_writer, tracing patterns) with LLMCore and ToxicityClassifierCore.
"""

import json
import time
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Self

import pydantic
from buttermilk._core.messages import AssistantMessage, SystemMessage, UserMessage
from opentelemetry import trace
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from buttermilk import logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_core import ProcessorCore, TraceParams
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import render_template
from buttermilk.utils.validators import import_class_from_path


class ClassifierResult(BaseModel):
    """Lightweight result structure for classifier operations.

    Mirrors LLMResult for consistency across processor types.
    """

    content: Any = Field(..., description="The classification output - structured object")
    metadata: dict[str, Any] = Field(default_factory=dict, description="API response, timing, model info")
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for correlation",
    )
    template_metadata: dict[str, Any] = Field(default_factory=dict, description="Template name, hash, etc")
    rendered_prompt: str = Field(default="", description="The rendered prompt/criteria sent to the classification API")
    error: str | None = Field(None, description="Error message if processing failed")


class ClassifierCore(ProcessorCore):
    """Stateless classification processor with template support.

    Extends ProcessorCore to share common infrastructure with LLMCore and
    ToxicityClassifierCore. Design: stateless, traceable, works directly
    with BaseRecord. No Agent inheritance - just process() and yield.

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

        context = ProcessingContext(session_id="classify", record=input_record)
        async for record in classifier.process(context):
            print(record.output)
        ```
    """

    model_config = {"extra": "forbid"}

    # Required fields
    template: str
    output_model: str | type[BaseModel]  # Will be converted to string path for serialization

    # Optional fields with defaults
    # Optional fields with defaults

    # Private attrs for runtime objects
    _resolved_output_model: type[BaseModel] | None = PrivateAttr(default=None)

    @field_validator("template")
    @classmethod
    def _validate_template(cls, v: str) -> str:
        """Validate that template is not empty."""
        if not v:
            raise ValueError("'template' is required for ClassifierCore")
        return v

    @field_validator("output_model")
    @classmethod
    def _validate_output_model(cls, v: str | type[BaseModel]) -> str | type[BaseModel]:
        """Validate that output_model is not None."""
        if v is None:
            raise ValueError("'output_model' is required for ClassifierCore")
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_output_model(cls, data: Any) -> Any:
        """Convert class to string path for storage."""
        if isinstance(data, dict) and "output_model" in data:
            output_model = data["output_model"]
            # If it's a class, convert to string path for storage
            if isinstance(output_model, type) and issubclass(output_model, BaseModel):
                data["output_model"] = f"{output_model.__module__}.{output_model.__name__}"
        return data

    @model_validator(mode="after")
    def _resolve_config(self) -> Self:
        """Resolve output_model string to class."""
        if isinstance(self.output_model, str) and self.output_model:
            try:
                self._resolved_output_model = import_class_from_path(self.output_model, expected_base_class=pydantic.BaseModel)
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"Failed to resolve output_model '{self.output_model}': {e}")
        elif isinstance(self.output_model, type) and issubclass(self.output_model, BaseModel):
            # If it's already a class (shouldn't happen after before validator, but handle it)
            self._resolved_output_model = self.output_model
        return self

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord through classification.

        Args:
            context: ProcessingContext with record to classify

        Yields:
            Any: The classification output directly (typed object or dict)
        """
        record = context.record
        processor_stage = context.session_id
        parent_trace_id = None
        tracer = trace.get_tracer("buttermilk.classifier")
        start_time = time.time()
        record_id = getattr(record, "record_id", "unknown")

        span_attributes = {
            "classifier.class": self.__class__.__name__,
            "classifier.template": self.template,
            "processor.stage": processor_stage,
            "record.id": record_id,
        }

        with tracer.start_as_current_span(f"classifier.{self.__class__.__name__}", attributes=span_attributes) as span:
            try:
                result = await self._classify_record(record)

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

                span.set_attribute("processing.time_ms", processing_time_ms)
                span.set_status(trace.Status(trace.StatusCode.OK))

                # Write ExecutionTrace to BigQuery (same pattern as LLMCore)
                # Build messages list: rendered_prompt as UserMessage, response as AssistantMessage
                response_content = json.dumps(output_content) if isinstance(output_content, dict) else str(output_content)
                trace_messages = [
                    UserMessage(content=result.rendered_prompt, source="classifier"),
                    AssistantMessage(content=response_content, source=self.__class__.__name__),
                ]

                # Emit success trace using inherited helper
                await self._emit_success_trace(
                    record=record,
                    outputs=output_content,
                    tp=TraceParams(
                        processor_stage=processor_stage,
                        parent_trace_id=parent_trace_id,
                        duration_ms=processing_time_ms,
                        messages=trace_messages,
                        inputs=None,
                        extra_metadata=stage_metadata,
                        execution_type="classification",
                        trace_id=result.trace_id,
                    ),
                )

                yield result.content

            except Exception as e:
                # Emit error trace using inherited helper
                await self._emit_error_trace(
                    record=record,
                    error=e,
                    tp=TraceParams(
                        processor_stage=processor_stage,
                        parent_trace_id=parent_trace_id,
                        duration_ms=int((time.time() - start_time) * 1000),
                        inputs=None,
                        execution_type="classification",
                    ),
                )

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise ProcessingError(f"Classification failed {processor_stage} for record {record_id}: {e}") from e

    async def _classify_record(self, record: BaseRecord, **kwargs: Any) -> ClassifierResult:
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
            # Render template: parameters (config) override inputs (record data)
            result = render_template(
                template=self.template,
                template_vars=self.parameters,  # Config overrides
                base_template_vars=inputs,  # Record data as base
                fail_on_unfilled=True,
            )
            rendered_text = result.rendered
            logger.debug(f"Classifier rendered template '{self.template}', unfilled vars: {result.unfilled_vars}, hash: {result.template_hash}")
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
            structured_output = self._map_to_schema(api_response, self._resolved_output_model)
            logger.debug(f"Classifier mapped to schema: {type(structured_output).__name__}")
        except Exception as e:
            raise ProcessingError(f"Failed to map response to {self._resolved_output_model.__name__}: {e}") from e

        return ClassifierResult(
            content=structured_output,
            metadata={"api_response": api_response},
            template_metadata={
                "name": self.template,
                "hash": result.template_hash,
                "unfilled_vars": list(result.unfilled_vars),
            },
            rendered_prompt=rendered_text,
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
    def _map_to_schema(self, response: dict[str, Any], schema: type[pydantic.BaseModel]) -> pydantic.BaseModel:
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
            model="openai/gpt-oss-safeguard-20b",
            output_model=SafeguardClassification,
        )
        ```
    """

    # Required field for HuggingFaceClassifier
    model: str

    # Private attr for LLM wrapper
    _llm_wrapper: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _initialize_llm_wrapper(self) -> Self:
        """Initialize LLM wrapper after validation."""
        from buttermilk import bm

        try:
            self._llm_wrapper = bm.llms.get_client(self.model)
            logger.debug(f"HuggingFaceClassifier initialized with model: {self.model}")
        except AttributeError as e:
            raise ValueError(f"Model '{self.model}' not found in LLM connections. Available: {list(bm.llms.connections.keys())}") from e
        return self

    @staticmethod
    def _parse_template_messages(rendered_text: str) -> list[tuple[str, str]]:
        """Parse rendered template into (role, content) tuples.

        Templates render with role prefixes like 'system:', 'user:', 'assistant:'.
        This parses them into structured messages.

        Args:
            rendered_text: Full rendered template with role prefixes

        Returns:
            List of (role, content) tuples
        """
        import re

        messages = []
        current_role = None
        current_content = []

        for line in rendered_text.strip().split("\n"):
            role_match = re.match(r"^(system|user|assistant):\s*", line, re.IGNORECASE)
            if role_match:
                # Save previous message if exists
                if current_role and current_content:
                    messages.append((current_role, "\n".join(current_content).strip()))
                current_role = role_match.group(1).lower()
                rest = line[role_match.end() :]
                current_content = [rest] if rest.strip() else []
            else:
                current_content.append(line)

        # Save last message
        if current_role and current_content:
            messages.append((current_role, "\n".join(current_content).strip()))

        return messages

    async def _classify(self, messages: list) -> pydantic.BaseModel | dict[str, Any]:
        """Call HuggingFace classification model via wrapper with structured output.

        Args:
            messages: List of LLM message objects

        Returns:
            Parsed Pydantic model if wrapper handled structured output,
            otherwise raw dict for manual mapping.
        """
        from buttermilk._core.llms import ModelOutput

        try:
            # Pass schema to wrapper - it will use structured output or function calling
            result = await self._llm_wrapper.create(
                messages=messages,
                schema=self._resolved_output_model,
            )

            # If wrapper returned a ModelOutput with parsed_object, use it directly
            if isinstance(result, ModelOutput) and result.parsed_object is not None:
                logger.debug(f"HuggingFace returned parsed {type(result.parsed_object).__name__} via structured output")
                return result.parsed_object

            # Fallback: parse raw content manually
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, str):
                    response = json.loads(result.content)
                else:
                    response = result.content

                logger.debug(f"HuggingFace response (manual parse): {response}")
                return response
            raise ProcessingError(f"Empty response from HuggingFace: {result}")

        except ProcessingError:
            raise
        except Exception as e:
            logger.error(f"HuggingFace classification failed: {e}")
            raise ProcessingError(f"HuggingFace classification failed: {e}") from e

    def _map_to_schema(self, response: dict[str, Any], schema: type[pydantic.BaseModel]) -> pydantic.BaseModel:
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

    async def _classify_record(self, record: BaseRecord, **kwargs: Any) -> ClassifierResult:
        """Override to parse template into proper LLM messages.

        Parses template with role prefixes (system:, user:, assistant:) into
        structured messages for the HuggingFace model.

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
            # Render template: parameters (config) override inputs (record data)
            result = render_template(
                template=self.template,
                template_vars=self.parameters,  # Config overrides
                base_template_vars=inputs,  # Record data as base
                fail_on_unfilled=True,
            )
            rendered_text = result.rendered
            logger.debug(
                f"HuggingFaceClassifier rendered template '{self.template}', unfilled vars: {result.unfilled_vars}, hash: {result.template_hash}"
            )
        except Exception as e:
            raise ProcessingError(f"Template rendering failed: {e}") from e

        # Step 2: Parse template into messages
        parsed_messages = self._parse_template_messages(rendered_text)

        # Convert to LLM message objects
        message_classes = {
            "system": SystemMessage,
            "user": UserMessage,
            "assistant": AssistantMessage,
        }
        messages = []
        for role, content in parsed_messages:
            msg_class = message_classes.get(role, UserMessage)
            if role == "user":
                messages.append(msg_class(content=content, source="user"))
            else:
                messages.append(msg_class(content=content, source=role))

        # Step 3: Call classification API
        try:
            api_response = await self._classify(messages)
            logger.debug(f"HuggingFaceClassifier received API response: {type(api_response).__name__}")
        except Exception as e:
            raise ProcessingError(f"HuggingFace API call failed: {e}") from e

        # Step 4: Use parsed result or map to schema
        if isinstance(api_response, self._resolved_output_model):
            # Wrapper already parsed and validated - use directly
            structured_output = api_response
            logger.debug(f"HuggingFaceClassifier using pre-parsed {type(structured_output).__name__}")
        else:
            # Fallback: manual mapping for raw dict responses
            try:
                structured_output = self._map_to_schema(api_response, self._resolved_output_model)
                logger.debug(f"HuggingFaceClassifier mapped to schema: {type(structured_output).__name__}")
            except Exception as e:
                raise ProcessingError(f"Failed to map response to {self._resolved_output_model.__name__}: {e}") from e

        # Store raw response for debugging (convert Pydantic to dict if needed)
        raw_response = api_response.model_dump() if hasattr(api_response, "model_dump") else api_response

        return ClassifierResult(
            content=structured_output,
            metadata={"api_response": raw_response},
            template_metadata={
                "name": self.template,
                "hash": result.template_hash,
                "unfilled_vars": list(result.unfilled_vars),
            },
            rendered_prompt=rendered_text,
        )


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

    # Private attr for API client config
    _client: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _initialize_client(self) -> Self:
        """Initialize Zentropi client with API credentials."""
        import os

        api_key = os.environ.get("ZENTROPI_API_KEY")
        if not api_key:
            raise ValueError("ZENTROPI_API_KEY environment variable is required")

        base_url = os.environ.get("ZENTROPI_BASE_URL", "https://api.zentropi.ai/v1/label")

        self._client = {"api_key": api_key, "base_url": base_url}
        logger.debug(f"ZentropiClassifier initialized with base_url: {base_url}")
        return self

    async def _classify(self, text: str, *, content: str) -> dict[str, Any]:
        """Call Zentropi API with criteria and content, with retry logic.

        Args:
            text: Rendered template containing classification criteria
            content: Text content to classify

        Returns:
            Dictionary with 'label', 'confidence', and 'compute_time'

        Raises:
            ValueError: If response is missing 'label' field
            aiohttp.ClientError: If API call fails after retries
        """
        import aiohttp

        from buttermilk._core.retry import RetryWrapper

        async def _make_request() -> dict[str, Any]:
            payload = {
                "content_text": content,
                "criteria_text": text,
            }

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self._client["base_url"],
                    headers={"Authorization": f"Bearer {self._client['api_key']}"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                response.raise_for_status()
                result = await response.json()

                if "label" not in result:
                    raise ValueError(f"Zentropi response missing 'label' field: {result.keys()}")

                return result

        # Use RetryWrapper for consistent retry logic across the codebase
        retry_wrapper = RetryWrapper(
            client=None,  # Not used, just need the retry config
            max_retries=3,
            min_wait_seconds=2.0,
            max_wait_seconds=30.0,
            jitter_seconds=2.0,
        )

        try:
            return await retry_wrapper._execute_with_retry(_make_request)
        except Exception as e:
            logger.error(f"Zentropi API call failed after retries: {e}")
            raise

    @staticmethod
    def _extract_criteria(rendered_text: str) -> str:
        """Extract criteria from rendered template, stripping role prefixes.

        Templates render with role prefixes like 'system:', 'user:', 'assistant:'.
        This method removes those prefixes and extracts just the criteria content,
        excluding the final user message (which contains the content placeholder).

        Args:
            rendered_text: Full rendered template with role prefixes

        Returns:
            Criteria text without role prefixes
        """
        import re

        lines = rendered_text.strip().split("\n")
        criteria_lines = []

        for line in lines:
            # Check for role prefix at start of line
            role_match = re.match(r"^(system|user|assistant):\s*", line, re.IGNORECASE)
            if role_match:
                # Add the rest of the line after the role prefix
                rest = line[role_match.end() :].strip()
                if rest:
                    criteria_lines.append(rest)
            else:
                # Continue with current section
                criteria_lines.append(line)

        # Join and clean up
        criteria = "\n".join(criteria_lines).strip()

        # Remove the "# Text to Classify" section and anything after (that's where content goes)
        text_to_classify_marker = "# Text to Classify"
        if text_to_classify_marker in criteria:
            criteria = criteria.split(text_to_classify_marker)[0].strip()

        return criteria

    async def _classify_record(self, record: BaseRecord, **kwargs: Any) -> ClassifierResult:
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
        from buttermilk.utils.templating import render_template

        # Extract content from record
        if not record.content:
            raise ValueError("record.content cannot be None or empty for classification")

        content = str(record.content)

        # Step 1: Render template (criteria only, without content placeholder)
        # We pass empty content to template so it renders criteria without duplicating content
        inputs = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        inputs.update(kwargs)
        inputs["content"] = ""  # Don't include content in template render

        try:
            # Render template: parameters (config) override inputs (record data)
            result = render_template(
                template=self.template,
                template_vars=self.parameters,  # Config overrides
                base_template_vars=inputs,  # Record data as base
                fail_on_unfilled=True,
            )
            rendered_text = result.rendered
            logger.debug(
                f"ZentropiClassifier rendered template '{self.template}', unfilled vars: {result.unfilled_vars}, hash: {result.template_hash}"
            )
        except Exception as e:
            raise ProcessingError(f"Template rendering failed: {e}") from e

        # Step 2: Parse template to extract criteria (strip role prefixes)
        criteria = self._extract_criteria(rendered_text)

        # Step 3: Call Zentropi API with criteria and content
        try:
            api_response = await self._classify(criteria, content=content)
            logger.debug(f"ZentropiClassifier received API response: {type(api_response).__name__}")
        except Exception as e:
            raise ProcessingError(f"Zentropi API call failed: {e}") from e

        # Step 4: Map to schema
        try:
            structured_output = self._map_to_schema(api_response, self._resolved_output_model)
            logger.debug(f"ZentropiClassifier mapped to schema: {type(structured_output).__name__}")
        except Exception as e:
            raise ProcessingError(f"Failed to map response to {self._resolved_output_model.__name__}: {e}") from e

        return ClassifierResult(
            content=structured_output,
            metadata={"api_response": api_response},
            template_metadata={
                "name": self.template,
                "hash": result.template_hash,
                "unfilled_vars": list(result.unfilled_vars),
            },
            rendered_prompt=criteria,
        )

    def _map_to_schema(self, response: dict[str, Any], schema: type[pydantic.BaseModel]) -> pydantic.BaseModel:
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
