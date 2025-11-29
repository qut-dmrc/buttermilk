# Import silence_logs early to suppress noisy task execution logs
from ..utils.silence_logs import silence_task_logs

# Re-apply silence to ensure logs are suppressed at the agent level
silence_task_logs()

from .classifier import ClassifierAgent, HuggingFaceClassifier, ZentropiClassifier
from .fetch import FetchAgent
from .judge import Judge
from .llm import LLMAgent
from .sheetexporter import GSheetExporter
from .spy import SpyAgent

ALL = [
    "ClassifierAgent",
    "FetchAgent",
    "GSheetExporter",
    "HuggingFaceClassifier",
    "Judge",
    "LLMAgent",
    "SpyAgent",
    "ZentropiClassifier",
]
