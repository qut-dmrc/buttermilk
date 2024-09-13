from google.cloud import language_v2

from typing import (
    Any,
)
from .toxicity import ToxicityModel, EvalRecord, Score
from promptflow.tracing import trace
from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from pathlib import Path
from typing import Literal
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from langchain_core.output_parsers import StrOutputParser

class GoogleModerate(ToxicityModel):
    model: str = "PaLM 2"
    standard: str = "Google Moderate Text v2"
    process_chain: str = "LanguageServiceClient"
    client: Any = None

    def init_client(self):
        client = language_v2.LanguageServiceClient()
        return client

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> language_v2.ModerateTextResponse:

        document = language_v2.Document(
            content=text,
            type_=language_v2.Document.Type.PLAIN_TEXT,
        )
        return self.client.moderate_text(document=document)

    def interpret(self, response: language_v2.ModerateTextResponse) -> EvalRecord:
        outcome = EvalRecord()
        try:
            outcome.predicted = False

            for category in response.moderation_categories:
                try:
                    # Google language_v2 sometimes returns a severity score
                    outcome.scores.append(Score(measure=category.name, confidence=category.confidence, severity=category.severity))
                except:
                    outcome.scores.append(Score(measure=category.name, confidence=category.confidence))
                if category.confidence > 0.5:
                    outcome.labels.append(category.name)
                    outcome.predicted = True


        except Exception as e:
            outcome.error = f"Unable to interpret result: {e}. {e.args}"
            outcome.response = str(response)

        return outcome
