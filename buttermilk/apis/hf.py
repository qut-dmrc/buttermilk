import os
from typing import Any, List, Optional, Union

import torch
from huggingface_hub import InferenceClient, login
from langchain.llms.base import LLM
from langchain.schema import SystemMessage
from langchain_experimental.chat_models import Llama2Chat
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.v1 import validator
from pyexpat import model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from datatools.gcloud import GCloud

MODIFIED_SYSTEM_PROMPT = """You are a helpful, accurate, and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501

gc = GCloud()
logger = gc.logger


class Llama2ChatMod(Llama2Chat):
    system_message: SystemMessage = SystemMessage(content=MODIFIED_SYSTEM_PROMPT)


class HFInferenceClient(LLM):
    hf_model_path: str
    options: dict[str, Any] = dict(
        do_sample=False,
        max_new_tokens=4096,
        repetition_penalty=0.1,
        top_p=0.7,
        temperature=1.0,
        watermark=False,
    )
    hf_client: Any = None

    @property
    def _llm_type(self) -> str:
        return f"hf-{self.hf_model_path}"

    @property
    def client(self):
        if self.hf_client is None:
            self.hf_client = self.get_client()
        return self.hf_client

    def get_client(self) -> Any:
        # access token with permission to access the model and PRO subscription
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
        return InferenceClient(self.hf_model_path)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

        res = self.client.text_generation(prompt, **self.options)
        return res.strip()


class HFTransformer(LLM):
    hf_model_path: str
    device: Union[str, torch.device] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device type (CPU or CUDA)",
    )
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    tokenizer: Any = Field(default=None)
    client: Any = Field(default=None)
    # pipeline: Optional[Any] = Field(default=None)
    initialized: bool = False
    torch_dtype: Optional[torch.dtype] = None

    # Field validator to dynamically set the 'device' based on CUDA availability
    @validator("device")  # v1 code
    @classmethod
    def validate_device(cls, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def _llm_type(self) -> str:
        return f"hf-{self.device}-{self.hf_model_path}"

    def _login(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)

    def _load_models(self):
        if self.initialized is False:
            # access token with permission to access the model and PRO subscription
            self._login()

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
            self.client = AutoModelForCausalLM.from_pretrained(
                self.hf_model_path, torch_dtype=self.torch_dtype, device_map=self.device
            )
            self.initialized = True

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        self._load_models()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[-1]
        output = self.client.generate(
            **inputs, max_new_tokens=kwargs.get("max_new_tokens", 100)
        )

        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        if not isinstance(result, str):
            logger.warning(
                f"DEBUG TODO: LLM Result for {self.name} is not a string: {result}"
            )
        else:
            result = result.lower().strip()

        return result


class HFPipeline(HFTransformer):
    def _load_models(self):
        if self.initialized is False:
            # access token with permission to access the model and PRO subscription
            login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

            self.pipeline = pipeline(
                "text-generation",
                model=self.name,
                device=self.device,
                dtype=torch.bfloat16,
            )

            self.initialized = True


class HFLlama270bn(HFInferenceClient):
    model_name = "meta-llama/Llama-2-70b-chat-hf"
