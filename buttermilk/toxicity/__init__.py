from typing import TypeVar,Any
from .aegis import Aegis
from .wildguard import Wildguard
from .google_moderate import GoogleModerate
from .nemo import NemoInputSimpleGPT4o,NemoInputComplexGPT4o,NemoOutputSimpleGPT4o, NemoOutputComplexGPT4o,NemoInputSimpleLlama31_70b,NemoInputComplexLlama31_70b,NemoOutputSimpleLlama31_70b,NemoOutputComplexLlama31_70b
from .toxicity import (Comprehend,
    Perspective,
    HONEST,
    LFTW,
    REGARD,
    AzureContentSafety,
    AzureModerator,
    OpenAIModerator,
    LlamaGuard1Replicate,
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Together,
    LlamaGuard3HF,
    LlamaGuard3HFInt8,
    LlamaGuard3Octo,
    GPTJT,
    LlamaGuard2HF,
    LlamaGuard3Local,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
    LlamaGuard2Local,
    LlamaGuard3LocalInt8,
    ShieldGemma,
    ToxicityModel)

TOXCLIENTS = [
    Comprehend,
    Perspective,
    AzureContentSafety,
    AzureModerator,
    OpenAIModerator,
    LlamaGuard1Replicate,
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Together,
    LlamaGuard3Octo,
    GoogleModerate,
    NemoInputSimpleGPT4o,NemoInputComplexGPT4o,NemoOutputSimpleGPT4o, NemoOutputComplexGPT4o,NemoInputSimpleLlama31_70b,NemoInputComplexLlama31_70b,NemoOutputSimpleLlama31_70b,NemoOutputComplexLlama31_70b
]

TOXCLIENTS_NOT_AVAILABLE = [
    LlamaGuard3HF,
    LlamaGuard3HFInt8,
    LlamaGuard2HF,
]
BAD_MODELS = []

TOXCLIENTS_LOCAL = [
    REGARD,
    LFTW,
    HONEST,
    LlamaGuard2Local,
    LlamaGuard3Local,
    LlamaGuard3LocalInt8,
    ShieldGemma,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
    GPTJT,
    Aegis,
    Wildguard
]

# Let's provide an interface for all the various toxicity models
def get_tox_model(model: str, **kwargs) -> ToxicityModel:
    return globals()[model]

def load_tox_model(model: str, **kwargs) -> ToxicityModel:
    return globals()[model](**kwargs)