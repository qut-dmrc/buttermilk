
from sys import flags
from pydantic import Field
import torch
from transformers import pipeline

import regex as re
from typing import (
    Any,
    ClassVar,
    Union,
)
from .toxicity import ToxicityModel, EvalRecord, Score, TEMPLATE_DIR
from promptflow.tracing import trace

from buttermilk.utils import read_text
class Wildguard(ToxicityModel):
    model: str = "allenai/wildguard"
    process_chain: str = "hf_transformers"
    standard: str = "wildguard"
    client: Any = None
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )
    options: ClassVar[dict] = dict(max_new_tokens=128,
            temperature=1.0, top_p=1.0)

    template: str = Field(default_factory=lambda: read_text(TEMPLATE_DIR / "wildguard.txt"))

    def init_client(self) -> None:
        self.client = pipeline("text-generation", model=self.model,device=self.device)

    @trace
    def make_prompt(self, content: str) -> str:
        prompt = self.template.format(content=content)
        return prompt

    @trace
    def call_client(
        self, prompt: str, **kwargs
    ) -> Any:

        response = self.client(prompt)
        return str(response[0]['generated_text']).strip()

    def interpret(self, response: Any) -> EvalRecord:
        ### Example output:
        # print(tokenizer.decode(result[0][len(tokenized_input['input_ids'][0]):], skip_special_tokens=True))
        # Harmful request: yes
        # Response refusal: yes
        # Harmful response: no

        labels = []

        outcome = EvalRecord()
        lines = response.split('\n')
        if len(lines) < 2:
            outcome.error = f"Unable to interpret result."
            outcome.response = response

        pattern = re.compile(r"^\s*([\w\s]*(request|response|refusal)):\s+(yes|no)", flags=re.IGNORECASE|re.MULTILINE)
        matches = pattern.findall(response)
        if not matches:
            outcome.error = f"Unable to interpret result."
            outcome.response = response
        else:
            outcome.predicted = False
            for match in matches:
                label = match[0]
                prediction = match[2]
                if prediction.lower().strip() == 'yes':
                    labels.append(label.lower().strip())
                    if label.lower().strip() in ["harmful request", "harmful response"]:
                        outcome.predicted = True

                    outcome.scores.append(
                        Score(measure=str(label), score=1.0, result=True)
                    )
                else:
                    outcome.scores.append(
                        Score(measure=str(label), score=0.0, result=False)
                    )


        return outcome
