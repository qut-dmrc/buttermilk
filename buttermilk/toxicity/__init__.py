from typing import Any, TypeVar

from buttermilk.toxicity.llamaguard import (
    LlamaGuard1Replicate,
    LlamaGuard1Together,
    LlamaGuard2HF,
    LlamaGuard2Local,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Local,
    LlamaGuard3LocalInt8,
    LlamaGuard3Together,
    MDJudge2,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
)

from .aegis import Aegis
from .google_moderate import GoogleModerate
from .nemo import (
    NemoInputComplexGPT4o,
    NemoInputComplexLlama31_70b,
    NemoInputSimpleGPT4o,
    NemoInputSimpleLlama31_70b,
    NemoOutputComplexGPT4o,
    NemoOutputComplexLlama31_70b,
    NemoOutputSimpleGPT4o,
    NemoOutputSimpleLlama31_70b,
)
from .toxicity import (
    GPTJT,
    HONEST,
    LFTW,
    REGARD,
    AzureContentSafety,
    AzureModerator,
    Comprehend,
    OpenAIModerator,
    Perspective,
    ShieldGemma,
    ShieldGemma2b,
    ShieldGemma9b,
    ToxicityModel,
)
from .wildguard import Wildguard

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
    GoogleModerate,
    REGARD,
    NemoInputSimpleGPT4o,NemoInputComplexGPT4o,NemoOutputSimpleGPT4o, NemoOutputComplexGPT4o,NemoInputSimpleLlama31_70b,NemoInputComplexLlama31_70b,NemoOutputSimpleLlama31_70b,NemoOutputComplexLlama31_70b
]


TOXCLIENTS_NOT_AVAILABLE = [
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
    ShieldGemma,ShieldGemma2b,ShieldGemma9b,
    MDJudge2,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
    GPTJT,
    Aegis,
    Wildguard
]

# Let's provide an interface for all the various toxicity models
def get_tox_flow(flow: str, **kwargs) -> ToxicityModel:
    return globals()[flow]

def load_tox_flow(flow: str, **kwargs) -> ToxicityModel:
    return globals()[flow](**kwargs)
