"""Configuration validator stub."""

from pathlib import Path

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    """Validation issue details."""

    component: str
    field: str
    message: str
    suggestion: str | None = None
    file_path: str | None = None


class ValidationReport(BaseModel):
    """Configuration validation report."""

    total_files_checked: int = 0
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    info: list[ValidationIssue] = []
    passed_checks: list[str] = []
    dependency_issues: list[str] = []
    is_valid: bool = True


def validate_configuration(config_path: str | Path) -> ValidationReport:
    """Validate configuration files.

    Args:
        config_path: Path to configuration directory

    Returns:
        ValidationReport with results
    """
    return ValidationReport(total_files_checked=0, is_valid=True, passed_checks=["Configuration validation not implemented"])
