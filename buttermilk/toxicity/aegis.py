
from enum import Enum, EnumMeta, IntEnum, StrEnum
from typing import (
    Any,
    AsyncGenerator,
    ClassVar,
    List,
    Literal,
    LiteralString,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pathlib import Path

from peft.peft_model import PeftModel
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from .toxicity import ToxicityModel, LlamaGuardTox, llamaguard_template, LlamaGuardTemplate, TEMPLATE_DIR
from buttermilk.utils import read_text

import numpy as np
import pandas as pd
from promptflow.tracing import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
import torch

class AegisCategories(Enum):
     O1 = "Violence"
     O2 = "Sexual"
     O3 = "Criminal Planning/Confessions"
     O4 = "Guns and Illegal Weapons"
     O5 = "Controlled/Regulated Substances"
     O6 = "Suicide and Self Harm"
     O7 = "Sexual (minor)"
     O8 = "Hate /identity hate"
     O9 = "PII/Privacy"
     O10 = "Harassment"
     O11 = "Threat"
     O12 = "Profanity"
     O13 = "Needs Caution"

class Aegis(LlamaGuardTox):
    model: str = "aegis"
    process_chain: str = "transformers_peft"
    standard: str = "aegis-defensive-1.0"
    client: Any = None
    template: str = Field(default_factory=lambda: read_text(TEMPLATE_DIR / "aegis.txt"))
    categories: EnumMeta = AegisCategories
    tokenizer: Any = None
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )
    options: ClassVar[dict] = {"max_new_tokens": 1024}

    def init_client(self):
        config = PeftConfig.from_pretrained("nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LlamaGuard-7b", revision="3e764390d6b39028ddea5b20603c89476107b41e")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b", revision="3e764390d6b39028ddea5b20603c89476107b41e", device_map=self.device, **self.options)
        self.client = PeftModel.from_pretrained(base_model, "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0", torch_device=self.device)
        return self.client

    @trace
    def call_client(
        self, text: str, **kwargs
    ) -> Any:
        content = self.make_prompt(text)
        inputs = self.tokenizer(
            content, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[-1]
        output = self.client.generate(
            **inputs, max_new_tokens=kwargs.get("max_new_tokens", 1000)
        )

        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return str(result)