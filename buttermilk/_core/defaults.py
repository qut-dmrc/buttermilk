from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
TEMPLATES_PATH = BASE_DIR / "templates"
BQ_SCHEMA_DIR = BASE_DIR / "schemas"

COL_PREDICTION = "prediction"


SLACK_MAX_MESSAGE_LENGTH = 3000
