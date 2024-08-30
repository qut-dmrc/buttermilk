from .aegis import Aegis
from .wildguard import Wildguard
from .toxicity import (Comprehend,
    Perspective,
    HONEST,
    LFTW,
    REGARD,
    AzureContentSafety,
    AzureModerator,
    OpenAIModerator,
    LlamaGuard1ReplicateAWQ,
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
    LlamaGuard1ReplicateAWQ,
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Together,
    LlamaGuard3HF,
    LlamaGuard3HFInt8,
    LlamaGuard3Octo,
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
    ShieldGemma,
    MDJudgeLocalDomain,
    MDJudgeLocalTask,
    GPTJT,
    Aegis,
    Wildguard
]