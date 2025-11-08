#!/usr/bin/env python
"""Profile buttermilk initialization speed.

This script measures initialization time at different stages to identify bottlenecks.
Run with: uv run python scripts/profile_init.py

Output shows:
- Import times for major modules
- Config loading time
- Infrastructure setup time
- Session creation time
- Total initialization time
"""

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """Context manager to time code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed * 1000:.2f}ms")


def profile_imports():
    """Profile import times for major buttermilk modules."""
    print("\n=== Import Profiling ===")

    with timer("Import asyncio"):
        import asyncio  # noqa: F401

    with timer("Import hydra"):
        import hydra  # noqa: F401

    with timer("Import omegaconf"):
        from omegaconf import DictConfig  # noqa: F401

    with timer("Import structlog"):
        import structlog  # noqa: F401

    with timer("Import pydantic"):
        import pydantic  # noqa: F401

    with timer("Import litellm"):
        import litellm  # noqa: F401

    with timer("Import anthropic"):
        import anthropic  # noqa: F401

    with timer("Import chromadb"):
        import chromadb  # noqa: F401

    with timer("Import autogen-core"):
        try:
            import autogen_core  # noqa: F401
        except ImportError:
            print("    (not installed)")

    with timer("Import buttermilk._core"):
        from buttermilk._core import config_bootstrap  # noqa: F401

    with timer("Import buttermilk.runner.cli"):
        from buttermilk.runner import cli  # noqa: F401


def profile_init():
    """Profile the full initialization flow with timing at each stage."""
    import asyncio

    print("\n=== Initialization Profiling ===")

    async def _profile_async():
        # Time config loading
        with timer("Config loading"):
            from buttermilk._core.config_bootstrap import (
                ConfigurationBootstrapper,
                resolve_config_dir,
            )

            config_dir = resolve_config_dir()
            bootstrapper = ConfigurationBootstrapper(
                config_path=config_dir,
                config_name="config",
                overrides=["++project_name=profile_test", "++job=init_profile"],
            )
            typed_config = bootstrapper.config

        # Time execution context creation
        with timer("Execution context setup"):
            from buttermilk._core.execution_context import from_config_async

            execution_context = await from_config_async(
                typed_config.infrastructure,
                project_name=typed_config.session.project_name,
            )

        # Time session creation
        with timer("Session creation"):
            from buttermilk._core.execution_context import (
                create_session_from_context_async,
            )

            bm = await create_session_from_context_async(
                execution_context=execution_context,
                session=typed_config.session,
                storage_configs=typed_config.storage,
                full_config=typed_config,
            )

        print(f"\n✓ Initialization complete - Session: {bm.session_info.session_id}")

        # Cleanup
        with timer("Graceful shutdown"):
            await bm.graceful_shutdown()

        return bm

    asyncio.run(_profile_async())


def profile_cold_start():
    """Profile a complete cold start using init_async."""
    import asyncio

    print("\n=== Cold Start Profiling (via init_async) ===")

    async def _cold_start():
        from buttermilk import init_async

        with timer("Full init_async()"):
            bm = await init_async(job="cold_start_profile", project_name="profile_test")

        print(f"✓ Session: {bm.session_info.session_id}")

        with timer("Shutdown"):
            await bm.graceful_shutdown()

    asyncio.run(_cold_start())


def main():
    """Run all profiling tests."""
    overall_start = time.perf_counter()

    print("=" * 60)
    print("Buttermilk Initialization Profiler")
    print("=" * 60)

    # Profile imports first (cold imports)
    profile_imports()

    # Profile initialization stages
    profile_init()

    # Profile complete cold start
    profile_cold_start()

    overall_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 60)
    print(f"Total profiling time: {overall_elapsed:.2f}s")
    print("=" * 60)

    print("\nRecommendations:")
    print("- Look for imports > 100ms - consider lazy loading")
    print("- Check if config loading > 500ms - optimize YAML parsing")
    print("- If execution context > 1000ms - check cloud API calls")
    print("- Use 'python -X importtime' for detailed import analysis")


if __name__ == "__main__":
    main()
