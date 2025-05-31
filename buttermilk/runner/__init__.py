# Import silence_logs early to suppress noisy task execution logs
from ..utils.silence_logs import silence_task_logs

# Re-apply silence to ensure logs are suppressed at the runner level
silence_task_logs()
