import os
from typing import Any

import torch
from huggingface_hub import InferenceClient, login
from langchain.llms.base import LLM
from langchain.schema import SystemMessage
from transformers import Pipeline, pipeline

MODIFIED_SYSTEM_PROMPT = """You are a helpful, accurate, and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501


class Llama2ChatMod:
    system_message: SystemMessage = SystemMessage(content=MODIFIED_SYSTEM_PROMPT)


class HFInferenceClient(LLM):
    hf_model_path: str
    options: dict[str, Any] = dict(
        do_sample=False,
        max_new_tokens=4096,
        repetition_penalty=0.1,
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
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        res = self.client.text_generation(prompt, **self.options)
        return res.strip()


def hf_pipeline(hf_model_path, **model_kwargs) -> Pipeline:
    # access token with permission to access the model
    login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
    if not (device := model_kwargs.pop("device", None)):
        device = "auto" if torch.cuda.is_available() else "cpu"
    max_new_tokens = model_kwargs.pop("max_new_tokens", 1000)
    client = pipeline(
        "text-generation",
        model=hf_model_path,
        device_map=device,
        max_new_tokens=max_new_tokens,
        model_kwargs=model_kwargs,
    )
    return client
