#!/usr/bin/env python3
"""Test Health Dashboard - Analyze and categorize test failures."""

import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TestFailure:
    """Represents a single test failure."""
    file: str
    test_name: str = ""
    error_type: str = ""
    error_message: str = ""
    category: str = ""


@dataclass
class TestHealth:
    """Overall test health statistics."""
    total_files: int = 0
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    collection_errors: List[TestFailure] = field(default_factory=list)
    failures_by_type: Dict[str, List[TestFailure]] = field(default_factory=lambda: defaultdict(list))
    failures_by_category: Dict[str, List[TestFailure]] = field(default_factory=lambda: defaultdict(list))


def categorize_error(error_msg: str) -> str:
    """Categorize error based on message patterns."""
    patterns = {
        "syntax_error": r"SyntaxError:|invalid syntax|'await' outside|unexpected",
        "import_error": r"ImportError:|cannot import name|No module named",
        "name_error": r"NameError:|name .* is not defined",
        "validation_error": r"ValidationError:|pydantic.*ValidationError|Field required",
        "type_error": r"TypeError:|got an unexpected keyword|takes .* positional",
        "attribute_error": r"AttributeError:|has no attribute|object has no attribute",
        "assertion_error": r"AssertionError:|assert.*==|assert.*!=",
        "file_not_found": r"FileNotFoundError:|No such file or directory",
        "key_error": r"KeyError:",
        "value_error": r"ValueError:",
        "timeout": r"TimeoutError:|timed out",
        "connection": r"ConnectionError:|Connection refused|Cannot connect",
        "mock_error": r"Mock.*Error|patch.*failed|MagicMock",
    }
    
    for category, pattern in patterns.items():
        if re.search(pattern, error_msg, re.IGNORECASE):
            return category
    return "other"


def get_test_category(file_path: str) -> str:
    """Determine test category from file path."""
    path_parts = Path(file_path).parts
    if "unit" in path_parts:
        return "unit"
    elif "integration" in path_parts:
        return "integration"
    elif "00initial" in str(file_path):
        return "initial"
    elif "agents" in path_parts:
        return "agents"
    elif "api" in path_parts:
        return "api"
    elif "data" in path_parts:
        return "data"
    elif "groupchat" in path_parts:
        return "groupchat"
    elif "tools" in path_parts:
        return "tools"
    elif "validation" in path_parts:
        return "validation"
    elif "examples" in path_parts:
        return "examples"
    elif "runner" in path_parts:
        return "runner"
    elif "storage" in path_parts:
        return "storage"
    elif "utils" in path_parts:
        return "utils"
    return "root"


def parse_pytest_output(output: str) -> TestHealth:
    """Parse pytest output to extract failure information."""
    health = TestHealth()
    
    # Parse summary line
    summary_match = re.search(r"(\d+) failed.*?(\d+) passed.*?(\d+) skipped.*?(\d+) warnings.*?(\d+) errors", output)
    if not summary_match:
        summary_match = re.search(r"(\d+) failed.*?(\d+) passed", output)
    
    if summary_match:
        health.failed = int(summary_match.group(1))
        health.passed = int(summary_match.group(2))
        if len(summary_match.groups()) > 2:
            health.skipped = int(summary_match.group(3)) if summary_match.group(3) else 0
            if len(summary_match.groups()) > 4:
                health.errors = int(summary_match.group(5)) if summary_match.group(5) else 0
    
    # Parse collection errors
    collection_error_blocks = re.findall(
        r"ERROR collecting (.*?)\n(.*?)(?=ERROR collecting|FAILED|=====|\Z)",
        output,
        re.DOTALL
    )
    
    for file_path, error_block in collection_error_blocks:
        failure = TestFailure(
            file=file_path.strip(),
            error_type="collection_error",
            error_message=error_block.strip()[:500],  # Truncate long messages
            category=get_test_category(file_path)
        )
        failure.error_type = categorize_error(error_block)
        health.collection_errors.append(failure)
        health.failures_by_type[failure.error_type].append(failure)
        health.failures_by_category[failure.category].append(failure)
    
    # Parse test failures
    failure_blocks = re.findall(
        r"FAILED (.*?)::(.*?) - (.*?)(?=FAILED|ERROR|=====|\Z)",
        output,
        re.DOTALL
    )
    
    for file_path, test_name, error_msg in failure_blocks:
        failure = TestFailure(
            file=file_path.strip(),
            test_name=test_name.strip(),
            error_message=error_msg.strip()[:500],
            category=get_test_category(file_path)
        )
        failure.error_type = categorize_error(error_msg)
        health.failures_by_type[failure.error_type].append(failure)
        health.failures_by_category[failure.category].append(failure)
    
    return health


def run_test_collection(test_path: str = "tests/") -> TestHealth:
    """Run pytest collection and basic test run to gather failure data."""
    print(f"Collecting test information from {test_path}...")
    
    # First, collect all tests (even with errors)
    cmd = ["uv", "run", "pytest", test_path, "--co", "-q"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    
    # Count total test files
    test_files = list(Path(test_path).rglob("test_*.py"))
    
    # Now run tests to get failure information (with timeout and limited output)
    cmd = ["uv", "run", "pytest", test_path, "--tb=short", "-q", "--timeout=5", "--timeout-method=thread"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    
    health = parse_pytest_output(result.stdout + result.stderr)
    health.total_files = len(test_files)
    health.total_tests = health.passed + health.failed + health.skipped + health.errors
    
    return health


def generate_report(health: TestHealth) -> str:
    """Generate a markdown report of test health."""
    report = []
    report.append("# Test Health Dashboard\n")
    report.append("## Summary Statistics\n")
    report.append(f"- **Total Test Files**: {health.total_files}")
    report.append(f"- **Total Tests**: {health.total_tests}")
    report.append(f"- **Passed**: {health.passed} ({health.passed*100/(health.total_tests or 1):.1f}%)")
    report.append(f"- **Failed**: {health.failed} ({health.failed*100/(health.total_tests or 1):.1f}%)")
    report.append(f"- **Errors**: {health.errors}")
    report.append(f"- **Skipped**: {health.skipped}")
    report.append(f"- **Collection Errors**: {len(health.collection_errors)}\n")
    
    # Failures by error type
    report.append("## Failures by Error Type\n")
    for error_type, failures in sorted(health.failures_by_type.items(), key=lambda x: -len(x[1])):
        report.append(f"### {error_type.replace('_', ' ').title()} ({len(failures)} issues)")
        
        # Show first 3 examples
        for failure in failures[:3]:
            report.append(f"- `{failure.file}`")
            if failure.test_name:
                report.append(f"  - Test: `{failure.test_name}`")
            report.append(f"  - Error: `{failure.error_message[:100]}...`")
        
        if len(failures) > 3:
            report.append(f"  - ... and {len(failures) - 3} more\n")
    
    # Failures by category
    report.append("\n## Failures by Test Category\n")
    for category, failures in sorted(health.failures_by_category.items(), key=lambda x: -len(x[1])):
        report.append(f"- **{category}**: {len(failures)} failures")
    
    # Priority fixes (collection errors and syntax errors)
    report.append("\n## Priority Fixes (Blocking Test Execution)\n")
    priority_issues = health.failures_by_type.get("syntax_error", []) + \
                     health.failures_by_type.get("import_error", []) + \
                     health.failures_by_type.get("name_error", [])
    
    if priority_issues:
        report.append("These issues prevent tests from even running:\n")
        for failure in priority_issues[:10]:
            report.append(f"1. `{failure.file}`")
            report.append(f"   - Type: {failure.error_type}")
            report.append(f"   - Error: `{failure.error_message[:150]}...`")
    else:
        report.append("No critical blocking issues found.\n")
    
    return "\n".join(report)


def save_detailed_json(health: TestHealth, output_file: str = "test_health_data.json"):
    """Save detailed test health data as JSON for tracking."""
    data = {
        "summary": {
            "total_files": health.total_files,
            "total_tests": health.total_tests,
            "passed": health.passed,
            "failed": health.failed,
            "errors": health.errors,
            "skipped": health.skipped,
            "collection_errors": len(health.collection_errors)
        },
        "failures_by_type": {
            error_type: [
                {
                    "file": f.file,
                    "test": f.test_name,
                    "error": f.error_message[:200]
                }
                for f in failures
            ]
            for error_type, failures in health.failures_by_type.items()
        },
        "failures_by_category": {
            cat: len(failures) for cat, failures in health.failures_by_category.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Detailed data saved to {output_file}")


def main():
    """Main entry point."""
    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/"
    
    health = run_test_collection(test_path)
    report = generate_report(health)
    
    # Save report
    report_file = "test_health_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to {report_file}")
    
    # Save detailed JSON
    save_detailed_json(health)
    
    # Print summary to console
    print("\n" + "="*60)
    print("TEST HEALTH SUMMARY")
    print("="*60)
    print(f"Pass Rate: {health.passed*100/(health.total_tests or 1):.1f}%")
    print(f"Critical Issues: {len(health.collection_errors)} collection errors")
    print("Top Issue Types:")
    for error_type, failures in sorted(health.failures_by_type.items(), key=lambda x: -len(x[1]))[:3]:
        print(f"  - {error_type}: {len(failures)} issues")
    print("="*60)


if __name__ == "__main__":
    main()
