import asyncio
import logging

from buttermilk._core.config_bootstrap import init_async
from buttermilk._core.storage_config import BigQueryStorageConfig
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.vertex import VertexBatchExecutor
from buttermilk.batch.processors import VertexBatchProcessor
from buttermilk.batch.runner import BatchPipelineRunner
from buttermilk.storage.bigquery import BigQueryStorage

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_verification():
    # 1. Initialize BM/Infrastructure (Mocking CLI init)
    # Ensure GCP connection - modify project/location as needed
    logger.info("Initializing infrastructure...")
    await init_async(project_name="verification", job="tja_batch_verification")

    # 2. Setup Components

    # Processor
    processor = VertexBatchProcessor(
        model="publishers/google/models/gemini-1.5-flash-001",
        template="judge",  # Assumes 'judge' template exists or load from path?
        # Note: Template loading might need 'session' context usually handled by runner.
        # For this script we might need to mock or ensure template registry is loaded.
        # Actually VertexBatchProcessor uses buttermilk.utils.templating.load_template
        # which looks in bm.session.template_paths if available.
        # Let's assume user has setup correct paths or we use absolute path/simple string for test.
        # Simplification: Use a dummy template string if file lookup fails, or ensure paths set.
        template_vars={"criteria": "Is this text trans-inclusive?"},
    )

    # Inject simple template directly if file loader fails?
    # Or rely on existing templates? tja/templates/judge.j2 exists?
    # Let's hope BM initialization covers it or we set it manually.

    # Executor
    executor = VertexBatchExecutor()

    # Storage (for results)
    storage_config = BigQueryStorageConfig(
        type="bigquery",
        project_id="prosocial-443205",
        dataset_id="testing",
        table_id="batch_runner_verification",
        full_table_id="prosocial-443205.testing.batch_runner_verification",
        schema_path="traces.schema.json",
        auto_create=True,
    )
    storage = BigQueryStorage(storage_config)

    # Source Records (Small Batch)
    source_records = [
        BaseRecord(record_id="tja-test-1", content="Trans rights are human rights."),
        BaseRecord(record_id="tja-test-2", content="This is a neutral sentence."),
        BaseRecord(record_id="tja-test-3", content="Harmful content warning."),
        BaseRecord(record_id="tja-test-4", content="Another test record."),
        BaseRecord(record_id="tja-test-5", content="Final test record."),
    ]

    # 3. Create Runner
    runner = BatchPipelineRunner(
        name="tja_batch_verification", source=source_records, batch_processor=processor, executor=executor, storage=storage, batch_size=5
    )

    # 4. Run
    logger.info("Starting Batch Pipeline Runner...")
    result = await runner.run()

    # 5. Report
    logger.info(f"Run completed with status: {result.status}")
    if result.output_uri:
        logger.info(f"Results saved to URI: {result.output_uri}")

    if result.error:
        logger.error(f"Run Error: {result.error}")

    # 6. Shutdown
    from buttermilk import bm

    await bm.graceful_shutdown()


if __name__ == "__main__":
    # We need to ensure authentication/BM context is ready.
    # Usually 'bm' lazy initializes.
    asyncio.run(run_verification())
