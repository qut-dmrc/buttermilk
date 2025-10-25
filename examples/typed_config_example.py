"""Example demonstrating NEW simplified typed configuration usage in Buttermilk.

This script shows the new simplified configuration structure with:
- Mode INSIDE run config (loaded via Hydra config groups like run=api)
- All execution params consolidated in run config
- Direct session access (no BMConfig wrapper)
- Unified limit parameter

This replaces loose dict access with type-safe code and provides a cleaner structure.
"""

from hydra import compose, initialize

from buttermilk._core.main_config import ButtermilkConfig, create_config_from_hydra
from buttermilk._core.run_config import RunMode


def example_basic_access():
    """Example: Basic type-safe configuration access with NEW structure."""
    print("\n=== Example 1: Basic Type-Safe Access (NEW STRUCTURE) ===")

    # Initialize Hydra and load configuration
    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        cfg = compose(config_name="config")

        # Convert to typed configuration
        typed_cfg: ButtermilkConfig = create_config_from_hydra(cfg)

        # Type-safe access with autocomplete
        print(f"Project: {typed_cfg.project_name}")
        print(f"Job: {typed_cfg.job}")
        print(f"Verbose: {typed_cfg.verbose}")
        print(f"Mode (inside run): {typed_cfg.run.mode}")  # NEW: mode inside run config

        # Direct session access (NEW: no wrapper)
        print(f"Session ID: {typed_cfg.session.session_id}")

        # Compare with old dict-based approach:
        # mode = cfg.run.get("mode", "console")  # May not exist, no autocomplete
        # session = cfg.bm.session_info  # Old: nested wrapper


def example_run_modes():
    """Example: Working with different run modes (NEW structure)."""
    print("\n=== Example 2: Run Mode Checking (NEW STRUCTURE) ===")

    # Console mode - load via run=console (config group)
    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        cfg = compose(config_name="config", overrides=["run=console"])
        typed_cfg = create_config_from_hydra(cfg)

        # NEW: Mode is inside run config (loaded from run=console)
        # Check mode with simple comparisons or match statement
        if typed_cfg.run.mode == RunMode.CONSOLE:
            print("Console mode")
            print(f"  Flow: {typed_cfg.run.flow}")  # NEW: in run config
            print(f"  Record ID: {typed_cfg.run.record_id}")

        elif typed_cfg.run.mode == RunMode.API:
            # NEW: API params in run config
            print(f"API mode: {typed_cfg.run.host}:{typed_cfg.run.port}")
            print(f"Workers: {typed_cfg.run.workers}")

        elif typed_cfg.run.mode in (RunMode.BATCH, RunMode.BATCH_RUN, RunMode.BATCH_ALL):
            print(f"Batch mode: {typed_cfg.run.mode}")
            # NEW: Unified limit parameter
            if typed_cfg.run.limit:
                print(f"  Limit: {typed_cfg.run.limit}")

        elif typed_cfg.run.mode == RunMode.PIPELINE:
            print("Pipeline mode")
            # NEW: Pipeline config in run
            pipeline = typed_cfg.run.pipeline
            if pipeline:
                print(f"  Concurrency: {pipeline.concurrency}")


def example_api_mode():
    """Example: API server configuration (NEW structure)."""
    print("\n=== Example 3: API Server Configuration (NEW STRUCTURE) ===")

    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        # Load API-specific configuration via config group
        # NEW: run=api loads conf/run/api.yaml with mode: api inside
        cfg = compose(config_name="config", overrides=["run=api", "run.port=8080"])
        typed_cfg = create_config_from_hydra(cfg)

        # NEW: Check mode inside run, access params from run config
        if typed_cfg.run.mode == RunMode.API:
            print("Starting API server...")
            print(f"  Host: {typed_cfg.run.host}")
            print(f"  Port: {typed_cfg.run.port}")
            print(f"  Workers: {typed_cfg.run.workers}")
            print(f"  Reload: {typed_cfg.run.reload}")
            print(f"  Log Level: {typed_cfg.run.log_level}")

            # Old approach would be:
            # mode = cfg.run.get("mode")  # Nested
            # host = cfg.get("host", "0.0.0.0")  # At root, not in run
            # port = int(cfg.get("port", 8000))  # Runtime type error if not int


def example_pipeline_mode():
    """Example: Pipeline configuration (NEW structure)."""
    print("\n=== Example 4: Pipeline Configuration (NEW STRUCTURE) ===")

    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        # Load pipeline configuration via config group
        cfg = compose(config_name="config", overrides=["run=pipeline"])
        typed_cfg = create_config_from_hydra(cfg)

        # NEW: Get pipeline configuration from run config
        pipeline = typed_cfg.run.pipeline
        if pipeline:
            print("Pipeline Configuration:")
            print(f"  Concurrency: {pipeline.concurrency}")
            print(f"  Limit: {pipeline.limit}")
            print(f"  Buffer Size: {pipeline.buffer_size}")
            print(f"  Flush Interval: {pipeline.flush_interval}s")

            # Get TMDB configuration
            tmdb = pipeline.get_tmdb_config()
            if tmdb:
                print(f"  TMDB Region: {tmdb.region}")

            # Get storage configurations
            source = pipeline.get_source_config()
            if source:
                print(f"  Source Type: {source.type}")
                print(f"  Source Dataset: {source.dataset_name}")

            output = pipeline.get_output_config()
            if output:
                print(f"  Output Type: {output.type}")


def example_storage_config():
    """Example: Working with storage configurations."""
    print("\n=== Example 5: Storage Configuration ===")

    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        cfg = compose(config_name="testing")
        typed_cfg = create_config_from_hydra(cfg)

        # Get specific storage configuration
        observations = typed_cfg.get_storage_config("observations")
        if observations:
            print("Observations Storage:")
            print(f"  Type: {observations.type}")
            print(f"  Dataset: {observations.dataset_name}")
            print(f"  Batch Size: {observations.batch_size}")

            # Type-specific access
            from buttermilk._core.storage_config import BigQueryStorageConfig

            if isinstance(observations, BigQueryStorageConfig):
                print(f"  Full Table ID: {observations.full_table_id}")
                print(f"  Project ID: {observations.project_id}")


def example_infrastructure():
    """Example: Infrastructure configuration."""
    print("\n=== Example 6: Infrastructure Configuration ===")

    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        cfg = compose(config_name="testing")
        typed_cfg = create_config_from_hydra(cfg)

        infra = typed_cfg.infrastructure

        # Cloud providers
        print("Cloud Providers:")
        for cloud in infra.clouds:
            print(f"  - {cloud.type}: {cloud.project_id} ({cloud.region})")

        # Tracing configuration
        print("\nTracing Configuration:")
        print(f"  Weave: {infra.tracing.weave.enabled}")
        print(f"  Traceloop: {infra.tracing.traceloop.enabled}")
        print(f"  OTEL: {infra.tracing.otel.enabled}")

        # Logging configuration
        if infra.logging:
            print("\nLogging:")
            print(f"  Type: {infra.logging.type}")
            print(f"  Verbose: {infra.logging.verbose}")
            print(f"  Console: {infra.logging.console}")


def example_validation():
    """Example: Configuration validation."""
    print("\n=== Example 7: Configuration Validation ===")

    from pydantic import ValidationError

    # This will fail validation
    try:
        invalid_config = ButtermilkConfig(
            verbose=True,
            project_name="test",
            job="example",
            run={"mode": "invalid_mode"},  # Invalid mode
            bm={
                "session_info": {
                    "project_name": "test",
                    "job": "example",
                }
            },
        )
    except ValidationError as e:
        print("Validation Error (as expected):")
        print(f"  {e.error_count()} errors found")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")

    # This will succeed
    try:
        valid_config = ButtermilkConfig(
            verbose=True,
            project_name="test",
            job="example",
            run={"mode": "console", "ui": "console"},  # Valid mode
            bm={
                "session_info": {
                    "project_name": "test",
                    "job": "example",
                }
            },
        )
        print("\nValid configuration created successfully!")
        print(f"  Mode: {valid_config.run.mode}")
    except ValidationError as e:
        print(f"Unexpected validation error: {e}")


def example_cli_overrides():
    """Example: Command-line overrides (NEW structure)."""
    print("\n=== Example 8: CLI Overrides (NEW STRUCTURE) ===")

    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        # Simulate command-line overrides
        # NEW: run=api loads config group, port in run config
        overrides = [
            "run=api",  # NEW: load run config group (includes mode: api)
            "run.port=9000",  # NEW: port in run config
            "verbose=true",
            "infrastructure.tracing.weave.enabled=true",
        ]

        cfg = compose(config_name="config", overrides=overrides)
        typed_cfg = create_config_from_hydra(cfg)

        print("Configuration with CLI overrides:")
        print(f"  Mode (inside run): {typed_cfg.run.mode}")  # NEW
        print(f"  Port (in run): {typed_cfg.run.port}")  # NEW
        print(f"  Verbose: {typed_cfg.verbose}")
        print(f"  Weave Tracing: {typed_cfg.infrastructure.tracing.weave.enabled}")
        print(f"  Session (direct): {typed_cfg.session.session_id}")  # NEW


def example_backward_compatibility():
    """Example: Backward compatibility with old config structure."""
    print("\n=== Example 9: Backward Compatibility ===")

    # Demonstrate that old config structure still works
    print("Old configs are automatically migrated:")
    print("  - bm.session_info → session")
    print("  - mode at root → run.mode")
    print("  - max_records/max_jobs → run.limit")
    print("  - Root-level params → run config")

    # Example: unified limit
    with initialize(version_base="1.3", config_path="../buttermilk/conf"):
        # Even if old config had max_jobs or max_records,
        # they're unified to run.limit
        cfg = compose(config_name="config", overrides=["run=batch"])
        typed_cfg = create_config_from_hydra(cfg)

        print("\nUnified limit parameter:")
        print(f"  Mode: {typed_cfg.run.mode}")
        print(f"  Limit: {typed_cfg.run.limit}")  # Works for both records and jobs


def main():
    """Run all examples."""
    examples = [
        example_basic_access,
        example_run_modes,
        example_api_mode,
        example_pipeline_mode,
        example_storage_config,
        example_infrastructure,
        example_validation,
        example_cli_overrides,
        example_backward_compatibility,
    ]

    print("=" * 70)
    print("BUTTERMILK CONFIGURATION EXAMPLES - NEW SIMPLIFIED STRUCTURE")
    print("=" * 70)
    print("\nKey improvements:")
    print("  - Mode inside run config (loaded via run=api config groups)")
    print("  - All execution params consolidated in run config")
    print("  - Direct session access (no BMConfig wrapper)")
    print("  - Unified limit parameter (replaces max_records/max_jobs)")
    print("  - Backward compatible with old structure")

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
