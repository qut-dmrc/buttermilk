from pathlib import Path

BASE_DIR = Path(__file__).absolute()
TEMPLATE_PATHS = [BASE_DIR.parent / "templates"]
BQ_SCHEMA_DIR = BASE_DIR.parent / "schemas"