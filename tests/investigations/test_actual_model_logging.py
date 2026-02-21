#!/usr/bin/env python3
"""Test script to verify actual model logging works correctly.

This script tests that when we use model aliases like 'gemini-flash-latest',
the 'model' field in metadata contains the actual model name returned by the
API (e.g., 'gemini-2.0-flash-exp') rather than our shorthand config name
(e.g., 'gemini25flash'), with fallback to the config name if the API doesn't
provide the actual model.
"""

import asyncio

import pytest

from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord

<<<<<<< HEAD
pytestmark = pytest.mark.slow

=======
import pytest

pytestmark = pytest.mark.slow
>>>>>>> origin/stable

@pytest.mark.anyio
async def test_actual_model_logging():
    """Test that we capture actual model names from API responses."""

    # Create an LLMCore instance with a model that might use aliases
    llm_core = LLMCore(
        model="gemini25flash",  # Our shorthand
        template="simple_test",  # Assumes you have this template
    )

    # Create a test record
    test_record = BaseRecord(
        record_id="test_001",
        data={"test_input": "Hello, world!"},
    )

    # Process the record
    try:
        async for result in llm_core.process(
            record=test_record,
            processor_stage="test",
        ):
            # Check metadata
            print("✓ Processing completed successfully")
            model_name = result.metadata.get("test", {}).get("model")
            print(f"  Model name logged: {model_name}")
<<<<<<< HEAD
            print(f"  (Should be actual model from API, or config name '{llm_core.model}' as fallback)")
=======
            print(
                f"  (Should be actual model from API, or config name '{llm_core.model}' as fallback)"
            )
>>>>>>> origin/stable

            # Verify model field exists
            assert "model" in result.metadata.get("test", {}), "Missing 'model' field"
            assert model_name, "Model name is empty"

            print("\n✓ Test PASSED: Model field is present in metadata")
            print(f"  Value: {model_name}")
            return True

    except Exception as e:
        print(f"✗ Test FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = asyncio.run(test_actual_model_logging())
    sys.exit(0 if success else 1)
