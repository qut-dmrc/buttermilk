from promptflow import tool

@tool
def get_criteria(filename: str) -> str:
    with open(filename, "r") as f:
            # For local files
            return f.read()