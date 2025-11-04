"""E2E test for complete image generation pipeline.

This test validates the full pipeline from CharacterPromptSource through
to CSV logging, using REAL components (no mocks).

TRUE E2E components:
- Real CharacterPromptSource with character data
- Real BatchExpansionProcessor
- Real ImageGenerationProcessor (calls Vertex AI!)
- Real GCSImageStorageProcessor (local storage for testing)
- Real CSVMetadataLogger

NO mocks except for external system boundaries we don't control in tests.
"""

import pandas as pd
import pytest

from buttermilk.agents.imagegen import VertexImagen3Fast, VertexImagen4Fast
from buttermilk.data.sources.character_prompt_source import CharacterPromptSource
from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.processors.batch_expansion import BatchExpansionProcessor
from buttermilk.processors.csv_metadata_logger import CSVMetadataLogger
from buttermilk.processors.gcs_image_storage import GCSImageStorageProcessor
from buttermilk.processors.image_generation import ImageGenerationProcessor


@pytest.mark.anyio
@pytest.mark.endtoend
async def test_image_generation_pipeline_end_to_end(real_bm, tmp_path):
    """Test complete image generation pipeline with real Vertex AI APIs.

    Pipeline: CharacterPromptSource → BatchExpansion → ImageGeneration → GCSStorage → CSVLogger

    Test Configuration:
    - 2 scenarios × 2 repetitions × 2 models = 8 images total
    - Uses CHEAP_IMAGE_CLIENTS only (cost control)
    - Local storage for testing (no actual GCS upload)

    Verifies:
    - All 8 images generated successfully (or acceptable failure rate)
    - Structured storage paths created correctly
    - CSV log contains all entries with correct schema
    - Session ID propagated throughout pipeline
    """
    # Create pipeline programmatically (simpler than Hydra for testing)
    # Note: session_id comes from real_bm.session_info.session_id automatically

    # 1. Source: Character-based prompts
    source = CharacterPromptSource(
        mask_attributes=["sexuality", "gender"],
        scenarios=["working in an office", "at a coffee shop"],
    )

    # 2. Processors: Complete pipeline stages
    processors = [
        # Stage 1: Expand to repetitions × models
        BatchExpansionProcessor(repetitions=2, models=[VertexImagen3Fast, VertexImagen4Fast]),
        # Stage 2: Generate images (real Vertex AI calls)
        # Model class is determined from record.metadata set by BatchExpansionProcessor
        ImageGenerationProcessor(),
        # Stage 3: Store images (local only for testing)
        GCSImageStorageProcessor(
            bucket="test-bucket",
            base_path="test-images",  # session_id added dynamically by processor
            local_only=True,  # Don't actually upload to GCS
            local_dir=str(tmp_path / "images"),
        ),
        # Stage 4: Log metadata to CSV (writes to local tmp_path, not GCS)
        CSVMetadataLogger(bucket="test-bucket", base_path=str(tmp_path)),
    ]

    # 3. Run pipeline
    orchestrator = PipelineOrchestrator(
        pipeline_name="test_image_gen_e2e",
        source=source,
        processors=processors,
        concurrency=2,
        limit=None,
    )

    processed_count = 0
    error_count = 0

    async for result in orchestrator():
        if result.metadata.get("error"):
            error_count += 1
        else:
            processed_count += 1

    # Note: finalize_processing is called automatically by PipelineOrchestrator

    # Verify results
    # Allow up to 25% failure rate (image gen APIs can be flaky)
    # Expected: 2 scenarios × 2 reps × 2 models = 8 images total
    assert processed_count >= 6, f"Expected at least 6/8 images, got {processed_count}"
    assert error_count <= 2, f"Too many failures: {error_count}/8"

    # Verify CSV log
    csv_path = tmp_path / "generation_log.csv"
    assert csv_path.exists(), "CSV log not created"

    df = pd.read_csv(csv_path)
    assert len(df) >= 6, f"CSV has {len(df)} rows, expected at least 6"

    # Verify CSV schema
    required_columns = [
        "prompt",
        "model",
        "timestamp",
        "filename",
        "scenario",
        "session_id",
        "repetition",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing CSV column: {col}"

    # Verify session ID consistency (it's set dynamically by bm)
    assert df["session_id"].nunique() == 1, "Multiple session IDs found"
    session_id = df["session_id"].iloc[0]  # Get the actual session ID used
    assert session_id, "Session ID should not be empty"

    # Verify models used are from CHEAP_IMAGE_CLIENTS
    models_used = df["model"].unique()
    assert all("VertexImagen" in m for m in models_used), f"Unexpected models: {models_used}"

    # Verify scenarios present
    scenarios = df["scenario"].unique()
    assert len(scenarios) == 2
    assert "working in an office" in scenarios
    assert "at a coffee shop" in scenarios

    # Verify image files exist
    image_dir = tmp_path / "images"
    image_files = list(image_dir.rglob("*.png"))
    assert len(image_files) >= 6, f"Expected at least 6 image files, found {len(image_files)}"

    # Verify structured paths (scenario/model/filename pattern)
    for image_file in image_files:
        # Path should be: images/test-images/{session_id}/{scenario}/{filename.png}
        parts = image_file.parts
        assert "images" in parts
        # Verify session ID in path
        assert any(session_id in part for part in parts)


@pytest.mark.anyio
@pytest.mark.endtoend
async def test_pipeline_with_single_character(real_bm, tmp_path):
    """Test pipeline with minimal configuration (single character, single model).

    This is a smoke test to verify basic pipeline functionality without
    hitting cost/quota limits.
    """
    # Minimal source: 1 scenario
    source = CharacterPromptSource(
        mask_attributes=["sexuality"],
        scenarios=["sitting at a desk"],
    )

    # Minimal processors: 1 repetition, 1 model
    processors = [
        BatchExpansionProcessor(repetitions=1, models=[VertexImagen3Fast]),
        ImageGenerationProcessor(),
        GCSImageStorageProcessor(
            bucket="test-bucket",
            base_path="test-single",  # session_id added dynamically
            local_only=True,
            local_dir=str(tmp_path / "images"),
        ),
        CSVMetadataLogger(bucket="test-bucket", base_path=str(tmp_path)),
    ]

    orchestrator = PipelineOrchestrator(
        pipeline_name="test_single_char",
        source=source,
        processors=processors,
        concurrency=1,
    )

    results = []
    async for result in orchestrator():
        results.append(result)

    # Note: finalize_processing is called automatically by PipelineOrchestrator

    # Verify at least one successful generation
    assert len(results) > 0, "No results generated"

    # Check CSV - find the generated CSV file (named by session_id)
    csv_files = list(tmp_path.glob("**/generation_log.csv"))
    assert len(csv_files) > 0, "No CSV file generated"
    csv_path = csv_files[0]

    df = pd.read_csv(csv_path)
    assert len(df) > 0
    # All records should have the same session_id (set by bm)
    assert df["session_id"].nunique() == 1, "Multiple session IDs found"
