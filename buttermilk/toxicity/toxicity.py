"""Toxicity classification processors for Buttermilk.

This module provides ToxicityClassifierCore (formerly ToxicityModel) and its
subclasses for toxicity/safety classification using various APIs and models.

ToxicityClassifierCore implements the Processor protocol for pipeline compatibility.
It uses EvalRecord as the standard output format for toxicity results.

Key classes:
- ToxicityClassifierCore: Base class for all toxicity classifiers
- ToxicityModel: Alias for ToxicityClassifierCore (backward compatibility)
- Perspective, Comprehend, AzureContentSafety, etc.: Specific implementations
"""

from __future__ import annotations

import abc
import asyncio
import os
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    ClassVar,
    Literal,
)

import pandas as pd
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    ExecutionTrace,
)  # Import AgentInput and ExecutionTrace
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable

from .types import EvalRecord, Score

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_torch_device() -> str:
    """Get the best available device for torch models.

    Returns 'cuda' if torch and CUDA are available, otherwise 'cpu'.
    Returns 'cpu' if torch is not installed.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


PerspectiveAttributes = Literal[
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
    "SEXUALLY_EXPLICIT",
]
PerspectiveAttributesExperimental = Literal[
    "TOXICITY_EXPERIMENTAL",
    "SEVERE_TOXICITY_EXPERIMENTAL",
    "IDENTITY_ATTACK_EXPERIMENTAL",
    "INSULT_EXPERIMENTAL",
    "PROFANITY_EXPERIMENTAL",
    "THREAT_EXPERIMENTAL",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
    # These are 'bridging' attributes: https://medium.com/jigsaw/announcing-experimental-bridging-attributes-in-perspective-api-578a9d59ac37
    "AFFINITY_EXPERIMENTAL",
    "COMPASSION_EXPERIMENTAL",
    "CURIOSITY_EXPERIMENTAL",
    "NUANCE_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
]


# Base class for all toxicity classifiers
class ToxicityClassifierCore(ProcessorCore):
    """Base class for toxicity/safety classification processors.

    Implements the Processor protocol for pipeline compatibility. Uses EvalRecord
    as the standard output format for toxicity classification results.

    Subclasses must implement:
    - init_client(): Initialize the API client
    - make_prompt(content): Format content for the API
    - interpret(response): Convert API response to EvalRecord

    Example:
        ```python
        class MyToxicityClassifier(ToxicityClassifierCore):
            model: str = "my-model"
            process_chain: str = "api"
            standard: str = "my-standard"

            def init_client(self):
                self._client = MyAPIClient()

            def make_prompt(self, content: str) -> str:
                return content

            def interpret(self, response: Any) -> EvalRecord:
                return EvalRecord(prediction=response["is_toxic"])
        ```
    """

    model: str
    process_chain: str
    standard: str
    info_url: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    options: ClassVar[dict] = {}
    call_options: ClassVar[dict] = {}

    # Private client attribute (not serialized)
    _client: Any = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_model(self) -> ToxicityClassifierCore:
        if self._client is None:
            self.init_client(**self.options)
            if self._client is None:
                raise ValueError(f"Unable to initialize client for {self.model}")
        return self

    def init_client(self) -> None:
        if self._client is None:
            raise NotImplementedError

    def _get_credential(self, key: str, required: bool = True) -> str | None:
        """Get credential from credentials dict, falling back to os.environ.

        Args:
            key: The credential key to retrieve
            required: If True, raise KeyError when key is missing from both sources

        Returns:
            The credential value, or None if not required and not found

        Raises:
            KeyError: If required=True and key not in credentials or environment
        """
        # Check credentials dict first
        if key in self.credentials:
            return self.credentials[key]

        # Fallback to environment variable
        if key in os.environ:
            return os.environ[key]

        if required:
            raise KeyError(f"{key} required in credentials dict or environment")
        return None

    def run(self, message: AgentInput) -> ExecutionTrace:  # Changed parameter name/type and return type
        # Assuming the AgentInput contains a record
        if not message.record:
            raise ValueError("AgentInput must contain a record for ToxicityModel.")

        # Process the record in the input message
        record = message.record

        response = self.moderate(content=record.content, record_id=record.record_id)  # Access content and record_id from Record
        if not isinstance(response, EvalRecord):
            raise ValueError(f"Expected an EvalRecord from toxicity model, got: {type(response)} for: {message}")

        # add identifying info in (already done in add_output_info)
        # response.model = self.model
        # response.process = self.process_chain
        # response.standard = self.standard
        # response.record_id = record.record_id # Already set in add_output_info

        # Create an ExecutionTrace object to return the results
        trace = ExecutionTrace(
            session_id=self.session_id,  # session_id is required for ExecutionTrace
            agent_info=self._config.model_dump(),  # agent_info is required for ExecutionTrace (as dict)
            inputs=message,  # Include the original input message
            outputs=response,  # Store the EvalRecord in outputs
            # Add other relevant metadata if needed
        )

        return trace  # Return the ExecutionTrace object

    def __call__(self, prompt: str, **kwargs) -> EvalRecord:
        return self.moderate(prompt=prompt, **kwargs)

    # @retry(
    #     retry=retry_if_exception_type(
    #         exception_types=(
    #             RateLimit,
    #             requests.exceptions.ConnectionError,
    #             urllib3.exceptions.ProtocolError,urllib3.exceptions.TimeoutError,
    #             OpenAIAPIConnectionError, OpenAIRateLimitError, AnthropicAPIConnectionError, AnthropicRateLimitError, ResourceExhausted
    #         ),
    #     ),
    #         # Wait interval: increasing exponentially up to a max of 30s between retries
    #         wait=wait_exponential_jitter(initial=1, max=30, jitter=5),
    #         # Retry up to five times before giving up
    #         stop=stop_after_attempt(5),
    # )

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> EvalRecord:
        return self._client.__call__(prompt, **self.call_options, **kwargs)

    @abc.abstractmethod
    def make_prompt(self, content):
        raise NotImplementedError

    def moderate_batch(self, dataset, **kwargs):  # -> Generator[Any, Any, None]:
        # if isinstance(self.client, transformers.Pipeline):
        #     # prepare batch
        #     dataset['text'] = dataset['text'].apply(self.make_prompt)
        #     input_ds = datasets.Dataset.from_pandas(dataset)
        #     for response in self.client(input_ds['text']):
        #         output = self.interpret(response)
        #         output = self.prepare_output_dict(output)
        #         yield output
        # el
        if isinstance(dataset, pd.DataFrame):
            for _, row in dataset.iterrows():
                # TODO: get this from the config instead
                record_id = row.get("id", row.get("record_id", row.get("name")))
                output = self.moderate(
                    content=row["content"],
                    record_id=record_id,
                    **kwargs,
                )
                output = output.model_dump()

                output = scrub_serializable(output)
                yield output
        else:
            for row in dataset:
                # TODO: get this from the config instead
                record_id = row.get("id", row.get("record_id", row.get("name")))
                output = self.moderate(
                    content=row["content"],
                    record_id=record_id,
                    **kwargs,
                )
                output = output.model_dump()

                output = scrub_serializable(output)
                yield output

    async def moderate_async(
        self,
        text: str,
        record_id: str,
        **kwargs,
    ) -> EvalRecord:
        return self.moderate(content=text, record_id=record_id, **kwargs)

    def moderate(
        self,
        content: str,
        record_id: str = None,
        **kwargs,
    ) -> EvalRecord:
        prompt = self.make_prompt(content)

        response = self.call_client(prompt=prompt, **kwargs)

        try:
            output = self.interpret(response)
        except ValueError as e:
            err_msg = f"Unable to interpret response from {self.model}. Error: {e} {e.args=}"
            output = EvalRecord(error=err_msg, response=response)
            logger.error(err_msg)
        output = self.add_output_info(output, record_id=record_id)
        return output

    @abc.abstractmethod
    def interpret(self, response: Any) -> EvalRecord:
        raise NotImplementedError

    def add_output_info(self, record: EvalRecord, record_id=None, **kwargs) -> EvalRecord:
        # add identifying info in
        record.model = self.model
        record.process = self.process_chain
        record.standard = self.standard
        if record_id is not None:
            record.record_id = record_id

        return record

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Processor protocol implementation for pipeline use.

        Wraps the synchronous moderate() method for use in async pipelines.
        Stores EvalRecord results in record.metadata[processor_stage].

        Consistent with ClassifierCore and LLMCore:
        - Uses trace_writer property for lazy loading
        - Builds standardized ExecutionTrace with agent_info, parameters, etc.
        - Stores results in record.metadata[processor_stage]

        Args:
            record: Input record to analyze for toxicity
            processor_stage: Pipeline stage name for metadata namespacing
            parent_trace_id: Optional parent trace ID for distributed tracing

        Yields:
            Record with toxicity results in metadata[processor_stage]

        Raises:
            ValueError: If record has no content
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())

        content = record.content
        if not content:
            raise ValueError(f"Record {record.record_id} has no content for toxicity analysis")

        # Wrap sync moderate() for async pipeline
        eval_record = await asyncio.to_thread(
            self.moderate,
            content=content,
            record_id=record.record_id,
        )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Emit execution trace using inherited helper (consistent with ClassifierCore/LLMCore)
        await self._emit_success_trace(
            record=record,
            outputs={
                "prediction": eval_record.prediction,
                "scores": [s.model_dump() for s in eval_record.scores],
                "labels": eval_record.labels,
                "error": eval_record.error,
            },
            processor_stage=processor_stage,
            parent_trace_id=parent_trace_id,
            duration_ms=duration_ms,
            inputs={
                "content": content[:500] if len(content) > 500 else content,
                "record_id": record.record_id,
            },
            extra_metadata={"eval_id": eval_record.eval_id},
            execution_type="toxicity_classification",
            trace_id=trace_id,
            parameters={
                "model": self.model,
                "process_chain": self.process_chain,
                "standard": self.standard,
            },
        )

        # Build output dict with prediction and optional labels
        output = {"prediction": eval_record.prediction}
        if eval_record.labels:
            output["labels"] = eval_record.labels
        if eval_record.scores:
            output["scores"] = [s.model_dump() for s in eval_record.scores]

        # Store full results in metadata (processor_stage namespaced)
        updated_metadata = record.metadata.copy() if record.metadata else {}
        updated_metadata[processor_stage] = {
            "classifier": self.__class__.__name__,
            "trace_id": trace_id,
            "processing_time_ms": int(duration_ms),
            "prediction": eval_record.prediction,
            "scores": [s.model_dump() for s in eval_record.scores],
            "labels": eval_record.labels,
            "model": eval_record.model,
            "standard": eval_record.standard,
            "process": eval_record.process,
            "eval_id": eval_record.eval_id,
            "error": eval_record.error,
        }

        yield record.model_copy(update={"output": output, "metadata": updated_metadata})


# Backward-compatible alias
ToxicityModel = ToxicityClassifierCore


class _HF(ToxicityClassifierCore):
    process_chain: str = "local transformers"
    model: str
    device: str | Any = Field(
        default_factory=_get_torch_device,
        description="Device type (CPU or CUDA or auto)",
    )
    options: ClassVar[dict] = dict(temperature=1.0)
    call_options: ClassVar[dict] = dict(max_new_tokens=128)
    tokenizer: Any = None

    def init_client(self) -> None:
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = self._get_credential("HUGGINGFACEHUB_API_TOKEN")

        login(token=token, new_session=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set a padding token

        self._client = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=True,
        ).to(self.device)

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        input_ids = self.tokenizer([prompt], padding="longest", return_tensors="pt").to(
            self.device,
        )["input_ids"]

        output = self._client.generate(
            input_ids=input_ids,
            **self.options,
            **self.call_options,
            **kwargs,
        )
        prompt_len = input_ids.shape[-1]
        response = self.tokenizer.decode(
            output[0][prompt_len:],
            skip_special_tokens=True,
        )
        try:
            result = response[0][0]["generated_text"].strip()
            return str(result[len(prompt) :])
        except:
            try:
                result = response.generations[0][0].text.strip()
                return str(result[len(prompt) :])
            except:
                result = response.strip()
                return result

    # def batch(self, texts: list[str]):
    #     inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    #     inputs = {key: val.to(model.device) for key, val in inputs.items()}


class Perspective(ToxicityClassifierCore):
    model: str = "perspective"
    process_chain: str = "api"
    standard: str = "perspective"

    def init_client(self) -> None:
        import google.auth
        from googleapiclient import discovery

        credentials, _ = google.auth.default()
        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            credentials=credentials,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def make_prompt(self, content: str) -> str:
        return content

    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(prediction=False)
        for key, value in response["attributeScores"].items():
            score = Score(measure=key, score=float(value["summaryScore"]["value"]))
            outcome.scores.append(score)
            if score.score and score.score > 0.5:
                outcome.labels.append(key)
                if key.lower() in [
                    "toxicity",
                    "severe_toxicity",
                    "toxicity_experimental",
                    "severe_toxicity_experimental",
                ]:
                    outcome.prediction = True

        return outcome

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        if not (attributes := kwargs.get("attributes")):
            # get all
            attributes = PerspectiveAttributes.__args__ + PerspectiveAttributesExperimental.__args__

        analyze_request = {
            "comment": {"text": prompt},
            "requestedAttributes": {attr: {} for attr in attributes},
            "doNotStore": True,
        }

        response = self._client.comments().analyze(body=analyze_request).execute()

        return response


class Comprehend(ToxicityClassifierCore):
    model: str = "comprehend"
    process_chain: str = "api"
    standard: str = "comprehend"

    def init_client(self) -> None:
        import boto3

        access_key = self._get_credential("AWS_ACCESS_KEY_ID")
        secret_key = self._get_credential("AWS_SECRET_ACCESS_KEY")
        region = self._get_credential("AWS_REGION")

        self._client = boto3.client(
            service_name="comprehend",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        return self._client.detect_toxic_content(
            LanguageCode="en",
            TextSegments=[{"Text": prompt}],
        )

    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            prediction=False,
        )
        result = response["ResultList"][0]

        outcome.scores = [Score(measure="Toxicity", score=result["Toxicity"])]
        if result["Toxicity"] > 0.5:
            outcome.labels.append("Toxicity")
            outcome.prediction = True
        else:
            outcome.prediction = False

        for score in result["Labels"]:
            new_score = Score(measure=score["Name"], score=score["Score"])
            outcome.scores.append(new_score)
            if new_score.score and new_score.score > 0.5:
                outcome.labels.append(score["Name"])

        return outcome


class AzureContentSafety(ToxicityClassifierCore):
    model: str = "AzureContentSafety"
    process_chain: str = "api"
    standard: str = "AzureContentSafety 2023-10-01"

    """ Azure Content Safety API: https://aka.ms/acs-doc
        https://contentsafety.cognitive.azure.com/
        Categories:
            Hate 	Hate and fairness-related harms refer to any content that attacks or uses pejorative or discriminatory language with reference to a person or identity group based on certain differentiating attributes of these groups including but not limited to race, ethnicity, nationality, gender identity and expression, sexual orientation, religion, immigration status, ability status, personal appearance, and body size.
            Sexual 	Sexual describes language related to anatomical organs and genitals, romantic relationships, acts portrayed in erotic or affectionate terms, pregnancy, physical sexual acts, including those portrayed as an assault or a forced sexual violent act against one's will, prostitution, pornography, and abuse.
            Violence 	Violence describes language related to physical actions intended to hurt, injure, damage, or kill someone or something; describes weapons, guns and related entities, such as manufacturers, associations, legislation, and so on.
            Self-harm 	Self-harm describes language related to physical actions intended to purposely hurt, injure, damage one's body or kill oneself.

        Severity levels (Text):
            The current version of the text model supports the full 0-7 severity scale. By default, the response will output 4 values: 0, 2, 4, and 6. Each two adjacent levels are mapped to a single level. Users could use "outputType" in request and set it as "EightSeverityLevels" to get 8 values in output: 0,1,2,3,4,5,6,7.

        **Note: We are using a cut-off of >=3 for the binary 'result' field.**
    """

    def init_client(self) -> None:
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.core.credentials import AzureKeyCredential

        API_KEY = self._get_credential("AZURE_CONTENT_SAFETY_KEY")
        ENDPOINT = self._get_credential("AZURE_CONTENT_SAFETY_ENDPOINT", required=False)
        if not ENDPOINT:
            ENDPOINT = "https://westus.api.cognitive.microsoft.com"

        credential = AzureKeyCredential(API_KEY)
        content_safety_client = ContentSafetyClient(ENDPOINT, credential)
        # blocklist_client = BlocklistClient(endpoint, credential)
        self._client = content_safety_client

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        from azure.ai.contentsafety.models import (
            AnalyzeTextOptions,
            AnalyzeTextOutputType,
        )

        request = AnalyzeTextOptions(
            text=prompt,
            output_type=AnalyzeTextOutputType.EIGHT_SEVERITY_LEVELS,
        )
        return self._client.analyze_text(request)

    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord(
            prediction=False,
        )

        for item in response.categories_analysis:
            measure = None
            score_labels = []
            severity_score = None
            try:
                measure = item.category
                severity = item.severity
                score_labels = [f"Severity: {severity}"]
                # arbitrary cutoff here. better to use the Score() value.
                if severity >= 3:
                    outcome.prediction = True
                    outcome.labels.append(measure)
                    score_labels.append(measure)

                severity_score = severity
            except Exception:
                raise ValueError(f"Unable to interpret Azure content safety score: {item}.")

            if measure is not None:
                outcome.scores.append(
                    Score(
                        measure=measure,
                        score=severity_score,
                        labels=score_labels,
                    ),
                )

        return outcome


class AzureModerator(ToxicityClassifierCore):
    model: str = "azure content-moderator"
    process_chain: str = "text-moderation-api"
    standard: str = "Azure Content Moderator"

    """ Azure Content Moderator screen text
        https://learn.microsoft.com/en-us/azure/ai-services/content-moderator/overview

        **Azure Content Moderator is deprecated as of February 2024 and will be retired by February 2027. It is replaced by Azure AI Content Safety, which offers advanced AI features and enhanced performance.**

        Terms: If the API detects any profane terms in any of the supported languages, those terms are included in the response. Profanity detection uses term-based matching with built-in list of profane terms in various languages

        Classification:
            Category1 refers to potential presence of language that may be considered sexually explicit or adult in certain situations.
            Category2 refers to potential presence of language that may be considered sexually suggestive or mature in certain situations.
            Category3 refers to potential presence of language that may be considered offensive in certain situations.
            Score is between 0 and 1. The higher the score, the higher the model is predicting that the category may be applicable. This feature relies on a statistical model rather than manually coded outcomes. We recommend testing with your own content to determine how each category aligns to your requirements.
            ReviewRecommended is either true or false depending on the internal score thresholds. Customers should assess whether to use this value or decide on custom thresholds based on their content policies.
    """

    def init_client(self) -> None:
        from azure.cognitiveservices.vision.contentmoderator import (
            ContentModeratorClient,
        )
        from msrest.authentication import CognitiveServicesCredentials

        SUBSCRIPTION_KEY = self._get_credential("AZURE_CONTENT_MODERATOR_KEY")
        ENDPOINT = self._get_credential("AZURE_CONTENT_MODERATOR_ENDPOINT", required=False)
        if not ENDPOINT:
            ENDPOINT = "https://westus.api.cognitive.microsoft.com"

        self._client = ContentModeratorClient(
            endpoint=ENDPOINT,
            credentials=CognitiveServicesCredentials(subscription_key=SUBSCRIPTION_KEY),
        )

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        text_stream = StringIO(prompt)
        response = self._client.text_moderation.screen_text(
            language="eng",
            text_content_type="text/plain",
            text_content=text_stream,
            autocorrect=True,
            pii=True,
            classify=True,
        ).as_dict()

        return response

    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord(
            prediction=False,
        )
        categories = {
            "category1": "sexually explicit or adult",
            "category2": "sexually suggestive or mature",
            "category3": "offensive",
        }

        for key, label in categories.items():
            score = Score(measure=label, score=response["classification"][key]["score"])

            outcome.scores.append(score)

            if score.score and score.score >= 0.5:
                outcome.labels.append(label)

        outcome.prediction = response["classification"]["review_recommended"]

        # Extra info usually includes  ["status", "language", r"tracking_id", "normalized_text", "auto_corrected_text"]
        # discard the ones we don't need.
        _result_keys = [
            "original_text",
            "classification",
            "normalized_text",
            "auto_corrected_text",
        ]
        outcome.metadata = {x: response[x] for x in response.keys() if x not in _result_keys}

        return outcome


class REGARD(ToxicityClassifierCore):
    model: str = "regard"
    process_chain: str = "evaluate"
    standard: str = "regard"

    def init_client(self) -> None:
        import evaluate

        device = _get_torch_device()
        if device == "cuda":
            self._client = evaluate.load(
                "regard",
                module_type="measurement",
                device="cuda",
            )
        else:
            self._client = evaluate.load("regard", module_type="measurement")

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        result = self._client.compute(data=[prompt])
        return result

    def interpret(self, response: Any) -> EvalRecord:
        result = response["regard"][0]

        # Load the message info into the output
        outcome = EvalRecord(
            prediction=False,
        )
        for record in result:
            score = Score(measure=record["label"], confidence=record["score"])
            outcome.scores.append(score)
            if record["score"] >= 0.5:
                outcome.labels.append(score.measure)
                if score.measure == "negative":
                    outcome.prediction = True

        return outcome


class HONEST(ToxicityClassifierCore):
    model: str = "honest"
    process_chain: str = "evaluate"
    standard: str = "honest"

    def init_client(self) -> None:
        import evaluate

        self._client = evaluate.load("honest", "en")

    def make_prompt(self, content: str) -> list[str]:
        return content.split(" ")

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        result = self._client.compute(predictions=prompt)
        return result

    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord()
        outcome.scores = [Score(measure="honest", score=response["honest_score"])]
        outcome.prediction = response["honest_score"] > 0.5
        if outcome.prediction:
            pass  # TODO: add label if we have it

        return outcome


class LFTW(ToxicityClassifierCore):
    model: str = "facebook/roberta-hate-speech-dynabench-r4-target"
    process_chain: str = "hf_transformers"
    standard: str = "lftw_r4_target"
    device: str | Any = Field(
        default_factory=_get_torch_device,
        description="Device type (CPU or CUDA or auto)",
    )
    options: ClassVar[dict] = dict()
    tokenizer: Any = None
    classes: dict = {}

    def init_client(self) -> None:
        from huggingface_hub import login
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        token = self._get_credential("HUGGINGFACEHUB_API_TOKEN")

        login(token=token, new_session=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set a padding token
        cfg = AutoConfig.from_pretrained(self.model)
        self.classes = cfg.id2label
        self._client = AutoModelForSequenceClassification.from_pretrained(self.model).to(
            self.device,
        )

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        import torch

        input_ids = self.tokenizer([prompt], return_tensors="pt").to(self.device)["input_ids"]
        with torch.no_grad():
            response = self._client(input_ids=input_ids, **self.options, **kwargs)
        logits = response.logits
        prediction_class_id = logits.argmax().item()
        result = self.classes[prediction_class_id]
        confidence = float(logits.softmax(dim=-1)[0][prediction_class_id])
        return dict(label=result, confidence=confidence)

    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord()
        outcome.scores = [
            Score(measure=response["label"], confidence=response["confidence"]),
        ]
        outcome.prediction = response["label"] == "hate"
        outcome.labels = [response["label"]]

        return outcome


class GPTJT(ToxicityClassifierCore):
    # Load model directly
    model: str = "togethercomputer/GPT-JT-Moderation-6B"
    process_chain: str = "hf_transformers"
    standard: str = "gpt-jt-mod-v1"
    template: str = Field(
        default_factory=lambda: read_text(TEMPLATE_DIR / "gpt-jt-mod-v1.txt"),
    )
    device: str | Any = Field(
        default_factory=_get_torch_device,
        description="Device type (CPU or CUDA)",
    )

    ResponseMap: dict[str, int] = {
        "casual": 1,
        "needs caution": 2,
        "need caution": 2,
        "needs intervention": 3,
        "possibly needs caution": 4,
        "probably needs caution": 5,
    }

    def init_client(self) -> None:
        from huggingface_hub import login

        from buttermilk.libs.hf import hf_pipeline

        token = self._get_credential("HUGGINGFACEHUB_API_TOKEN")

        login(token=token, new_session=False)
        self._client = hf_pipeline(
            hf_model_path="togethercomputer/GPT-JT-Moderation-6B",
            device=self.device,
            max_new_tokens=3,
        )

    def make_prompt(self, content: str) -> str:
        prompt = self.template.format(content=content)
        return prompt

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        response = self._client(prompt)

        if len(response) > 1:
            raise ValueError("Expected only one result from model")
        result = response[0]["generated_text"]
        result = result.removeprefix(prompt)

        return result.strip()

    def interpret(self, response: Any) -> EvalRecord:
        # Load the message info into the output
        outcome = EvalRecord()
        try:
            outcome.response = str(response)
            outcome.scores = [
                Score(
                    measure=self.standard,
                    score=self.ResponseMap[response],
                    labels=[response],
                ),
            ]

            outcome.prediction = self.ResponseMap[response] >= 2
            outcome.labels = [response]
        except Exception as e:
            raise ValueError(f"Unable to interpret response from GPT-JT model. {response=}, {e=}, {e.args=}")

        return outcome


###
# TODO: add https://github.com/unitaryai/detoxify
####


class OpenAIOmni(ToxicityClassifierCore):
    model: str = ""


class OpenAIModerator(ToxicityClassifierCore):
    model: str = "text-moderation-latest"
    process_chain: str = "api"
    standard: str = "openaimod"

    def init_client(self) -> None:
        import openai

        openai.api_type = "openai"
        self._client = openai.moderations

    def make_prompt(self, content: str) -> str:
        return content

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        return self._client.create(input=prompt, model=self.model)

    def interpret(self, response: Any) -> EvalRecord:
        if len(response.results) > 1:
            raise ValueError("Expected only one result from OpenAI model")
        result = response.results[0].dict()

        # Load the message info into the output
        outcome = EvalRecord()
        outcome.scores = [Score(measure=k, score=v) for k, v in result["category_scores"].items()]

        outcome.prediction = result["flagged"]
        outcome.labels = [c for c, v in result["categories"].items() if v]

        return outcome


class ShieldGemma(ToxicityClassifierCore):
    model: str = "google/shieldgemma-27b"
    process_chain: str = "local transformers"
    standard: str = "shieldgemma"
    tokenizer: Any = None
    classes: Any = None
    _tpl: str = ""
    _criteria: str = ""
    criteria: str
    device: str | Any = Field(
        default_factory=_get_torch_device,
        description="Device type (CPU or CUDA or auto)",
    )

    def init_client(self) -> None:
        import torch
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = self._get_credential("HUGGINGFACEHUB_API_TOKEN")

        login(token=token, new_session=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._client = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        vocab = self.tokenizer.get_vocab()
        self.classes = [vocab["Yes"], vocab["No"]]

        template = read_yaml(TEMPLATE_DIR / "shieldgemma.yaml")
        self._criteria = template["criteria"][self.criteria]
        self._tpl = template["template"]

    def make_prompt(self, text):
        prompt = self._tpl.format(text=text, criteria=self._criteria)
        return prompt

    def call_client(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            model_outputs = self._client(**inputs)

        logits = model_outputs.logits

        selected_logits = logits[0, -1, self.classes]

        # Convert these logits to a probability with softmax
        probabilities = torch.softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = float(probabilities[0].item())

        return dict(score=score)

    def interpret(self, response: Any) -> EvalRecord:
        outcome = EvalRecord(
            prediction=False,
        )

        score = response["score"]
        outcome.scores = [Score(measure="GemmaGuardDefault", score=score)]
        if score > 0.5:
            outcome.labels.append("violating")
            outcome.prediction = True
        else:
            outcome.prediction = False

        return outcome


class ShieldGemma2b(ShieldGemma):
    model: str = "google/shieldgemma-2b"


class ShieldGemma9b(ShieldGemma):
    model: str = "google/shieldgemma-9b"


class ToxicChat(ToxicityClassifierCore):
    model: str = "toxicchat"
    process_chain: str = "hf-api"
    standard: str = "toxicchat"

    def init_client(self) -> None:
        pass

    def interpret(self, response, **kwargs) -> EvalRecord:
        return EvalRecord(**response)


class Zentropi(ToxicityClassifierCore):
    """Zentropi labeling API wrapper.

    Zentropi provides a classification API for content labeling.
    This class adapts the Zentropi API response format to the ToxicityModel interface.

    The API requires:
        - content_text: The text to be labeled
        - criteria_text: The labeling criteria (passed as the system message/template)

    Expected Zentropi response format:
    {
        "label": "1",
        "confidence": 0.87,
        "compute_time": 0.324
    }
    """

    model: str = "cope-latest"
    process_chain: str = "api"
    standard: str = "zentropi"
    criteria: str = ""  # The labeling criteria (system message/template)

    def init_client(self) -> None:
        """Initialize client with credentials from credentials dict or environment variables.

        Requires:
            ZENTROPI_API_KEY: API key for Zentropi service (from credentials or env var)
            ZENTROPI_BASE_URL: (optional) API endpoint, defaults to https://api.zentropi.ai/v1/label
        """
        api_key = self._get_credential("ZENTROPI_API_KEY")
        base_url = self._get_credential("ZENTROPI_BASE_URL", required=False)
        if not base_url:
            base_url = "https://api.zentropi.ai/v1/label"

        self._client = {
            "api_key": api_key,
            "base_url": base_url,
        }

    def make_prompt(self, content: str) -> str:
        """Pass content through unchanged."""
        return content

    def interpret(self, response: dict[str, Any]) -> EvalRecord:
        """Convert Zentropi API response to EvalRecord.

        Args:
            response: Zentropi API response containing:
                - label (str): The classification label
                - confidence (float): Confidence score for the label
                - compute_time (float): Processing time in seconds

        Returns:
            EvalRecord with prediction, scores, and labels

        Raises:
            ValueError: If required 'label' field is missing from response
        """
        if "label" not in response:
            raise ValueError(f"Zentropi response missing required 'label' field. Got: {response.keys()}")

        # Extract label and determine prediction
        label = response["label"]
        confidence = response.get("confidence", 0.0)

        # Create score from confidence
        scores = [Score(measure="confidence", score=confidence)]

        # Add compute_time as metadata in a score if present
        if "compute_time" in response:
            scores.append(Score(measure="compute_time", score=response["compute_time"]))

        # Prediction is True if label indicates positive classification
        # Label "1" or truthy string values indicate positive
        prediction = label in ("1", "true", "True", "yes", "Yes", True)

        return EvalRecord(
            prediction=prediction,
            scores=scores,
            labels=[label] if label else [],
        )

    def call_client(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Call Zentropi API with content and criteria.

        Args:
            prompt: Text content to classify (content_text)

        Returns:
            Zentropi API response dict

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If criteria is not set
        """
        import requests

        if not self.criteria:
            raise ValueError("Zentropi requires criteria to be set. Pass the system message/template as the 'criteria' field.")

        payload = {
            "content_text": prompt,
            "criteria_text": self.criteria,
        }

        # Add model if not default
        if self.model and self.model != "cope-latest":
            payload["model"] = self.model

        response = requests.post(
            self.client["base_url"],
            headers={"Authorization": f"Bearer {self.client['api_key']}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
