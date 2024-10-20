from typing import TypeVar,Any

from buttermilk.toxicity.llamaguard import (LlamaGuard1Replicate,
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Together,
    LlamaGuard2HF,
    LlamaGuard3Local,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
    MDJudge2,
    LlamaGuard2Local,
    LlamaGuard3LocalInt8)
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
    GPTJT,
    ShieldGemma,ShieldGemma2b,ShieldGemma9b,
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