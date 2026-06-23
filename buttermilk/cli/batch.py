"""CLI commands for batch processing operations.

These commands allow submitting, monitoring, and fetching batch jobs
without requiring a continuous process. Jobs can be managed from any
machine with GCS access.

Usage:
    bm batch submit <config>      Submit batch job, print job_id, exit
    bm batch status <job_id>      Check job status
    bm batch fetch <job_id>       Fetch and save results
    bm batch list                 List recent batch jobs from save_dir
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

import click

from buttermilk import logger


@click.group()
def batch() -> None:
    """Batch processing commands for managing async Vertex AI batch jobs.

    Submit jobs, check status, and fetch results without requiring
    a continuous process. Works from any machine with GCS access.
    """


@batch.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def submit(config: Path, json_output: bool) -> None:
    """Submit a batch job from a config file.

    CONFIG is a YAML/JSON config file specifying the batch job parameters.
    Prints the job_id which can be used with status/fetch commands.

    Example:
        bm batch submit batch_config.yaml
    """
    import asyncio

    import yaml  # type: ignore[import-untyped]  # PyYAML: no stubs available

    from buttermilk import init_async

    click.echo("Loading configuration...", err=True)

    # Load config file
    config_content = config.read_text()
    if config.suffix in (".yaml", ".yml"):
        batch_config = yaml.safe_load(config_content)
    else:
        batch_config = json.loads(config_content)

    # Initialize buttermilk
    async def run_submit() -> dict:
        bm = await init_async(job="batch-submit")

        from buttermilk.batch.managers import BatchJobManager, BatchRequest

        manager = BatchJobManager(client=bm.genai)

        # Extract batch parameters from config
        model = batch_config.get("model", "gemini-2.5-flash")
        requests_data = batch_config.get("requests", [])

        # Convert raw request dicts to BatchRequest objects
        requests = [BatchRequest(**r) for r in requests_data]

        if not requests:
            raise ValueError("No requests found in config file")

        # Submit the batch job
        job = await manager.submit_batch(
            model=model,
            requests=requests,
        )

        # Extract job_id from the internal tracking
        # The job_id is the last part of the job name
        job_id = None
        for jid, info in manager._active_jobs.items():
            if info["job"].name == job.name:
                job_id = jid
                break

        return {
            "job_id": job_id,
            "vertex_job_name": job.name,
            "model": model,
            "request_count": len(requests),
            "save_dir": bm.session_info.save_dir,
        }

    try:
        result = asyncio.run(run_submit())

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("Job submitted successfully!")
            click.echo(f"  Job ID: {result['job_id']}")
            click.echo(f"  Vertex Job: {result['vertex_job_name']}")
            click.echo(f"  Model: {result['model']}")
            click.echo(f"  Requests: {result['request_count']}")
            click.echo(f"\nTo check status: bm batch status {result['job_id']}")
            click.echo(f"To fetch results: bm batch fetch {result['job_id']}")

    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}")
        if json_output:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@batch.command()
@click.argument("job_id")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option(
    "--save-dir",
    type=str,
    help="GCS save directory (auto-detected from session if not provided)",
)
@click.option(
    "--search",
    "-s",
    is_flag=True,
    help="Search across all sessions in the bucket to find the manifest",
)
def status(job_id: str, json_output: bool, save_dir: str | None, search: bool) -> None:
    """Check the status of a batch job.

    JOB_ID is the job identifier returned by the submit command.

    Example:
        bm batch status batch_abc123def456
        bm batch status batch_abc123def456 --search
    """
    import asyncio

    from buttermilk import init_async

    async def run_status() -> dict:
        bm = await init_async(job="batch-status")

        from buttermilk.batch.managers import BatchJobManager

        manager = BatchJobManager(client=bm.genai)

        # Override save_dir if provided
        if save_dir:
            bm.session_info.save_dir = save_dir

        return manager.get_job_status(job_id, search=search)

    try:
        result = asyncio.run(run_status())

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Job Status: {job_id}")
            click.echo(f"  State: {result['state']}")
            click.echo(f"  Complete: {'Yes' if result['is_complete'] else 'No'}")
            if result["is_complete"]:
                click.echo(f"  Success: {'Yes' if result['is_success'] else 'No'}")
            if result.get("error"):
                click.echo(f"  Error: {result['error']}")

    except FileNotFoundError:
        msg = f"Job manifest not found for: {job_id}"
        if json_output:
            click.echo(json.dumps({"error": msg, "job_id": job_id}))
        else:
            click.echo(f"Error: {msg}", err=True)
            click.echo("Make sure you're using the correct save_dir or session.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        if json_output:
            click.echo(json.dumps({"error": str(e), "job_id": job_id}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@batch.command()
@click.argument("job_id")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option(
    "--save-dir",
    type=str,
    help="GCS save directory (auto-detected from session if not provided)",
)
@click.option(
    "--search",
    "-s",
    is_flag=True,
    help="Search across all sessions in the bucket to find the manifest",
)
def fetch(job_id: str, json_output: bool, output: str | None, save_dir: str | None, search: bool) -> None:
    """Fetch results from a completed batch job.

    JOB_ID is the job identifier returned by the submit command.
    If the job is still running, returns status instead of results.

    Example:
        bm batch fetch batch_abc123def456
        bm batch fetch batch_abc123def456 --search
        bm batch fetch batch_abc123def456 -o results.json
    """
    import asyncio

    from buttermilk import init_async

    async def run_fetch() -> dict[str, Any]:
        bm = await init_async(job="batch-fetch")

        from buttermilk.batch.managers import BatchJobManager

        manager = BatchJobManager(client=bm.genai)

        # Override save_dir if provided
        if save_dir:
            bm.session_info.save_dir = save_dir

        return manager.fetch_results(job_id, search=search)

    try:
        result = asyncio.run(run_fetch())

        # Check if result is a success dict (with summary and results) or status/error dict
        if isinstance(result, dict) and "summary" in result and "results" in result:
            summary = result["summary"]
            results_data = result["results"]

            if output:
                output_path = Path(output)
                output_path.write_text(json.dumps(results_data, indent=2))
                click.echo(f"Results saved to: {output_path}")
                click.echo(f"Total results: {len(results_data)}")
            elif json_output:
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo("=" * 40)
                click.echo("Batch Processing Summary")
                click.echo(f"  Job ID: {summary['job_id']}")
                click.echo(f"  Model: {summary['model']}")
                click.echo(f"  Requests: {summary['request_count']}")
                click.echo(f"  Success: {summary['success_count']}")
                click.echo(f"  Errors: {summary['error_count']}")
                if summary.get("total_cost_usd") is not None:
                    click.echo(f"  Estimated Cost: ${summary['total_cost_usd']:.4f}")
                click.echo("=" * 40)

                click.echo(f"\nFetched {len(results_data)} results")
                click.echo("\nFirst 3 results:")
                for i, r in enumerate(results_data[:3]):
                    click.echo(f"\n  [{i + 1}] record_id: {r['record_id']}")
                    if r.get("response"):
                        response_preview = r["response"][:200]
                        if len(r["response"]) > 200:
                            response_preview += "..."
                        click.echo(f"      response: {response_preview}")
                    if r.get("error"):
                        click.echo(f"      error: {r['error']}")
                    if r.get("cost_usd") is not None:
                        click.echo(f"      cost: ${r['cost_usd']:.6f}")

                if len(results_data) > 3:
                    click.echo(f"\n  ... and {len(results_data) - 3} more results")
                click.echo("\nUse -o <file> to save all results to a file")
        # Status or error dict (not processed yet)
        elif json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            status_val = result.get("status", result.get("state", "unknown"))
            click.echo(f"Job {job_id}: {status_val}")
            if result.get("error"):
                click.echo(f"  Error: {result['error']}")
            elif result.get("request_count"):
                click.echo(f"  Requests: {result['request_count']}")
                click.echo("\nJob is still running. Check back later.")

    except FileNotFoundError:
        msg = f"Job manifest not found for: {job_id}"
        if json_output:
            click.echo(json.dumps({"error": msg, "job_id": job_id}))
        else:
            click.echo(f"Error: {msg}", err=True)
            click.echo("Make sure you're using the correct save_dir or session.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        if json_output:
            click.echo(json.dumps({"error": str(e), "job_id": job_id}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@batch.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--limit", "-n", default=10, help="Maximum number of jobs to list")
@click.option(
    "--save-dir",
    type=str,
    help="GCS save directory (auto-detected from session if not provided)",
)
def list_jobs(json_output: bool, limit: int, save_dir: str | None) -> None:
    """List recent batch jobs from the save directory.

    Shows job IDs, submission times, and current status.

    Example:
        bm batch list
        bm batch list -n 20
    """
    import asyncio

    from cloudpathlib import AnyPath, CloudPath

    from buttermilk import init_async
    from buttermilk.batch.managers import BatchJobManifest

    async def run_list() -> list[dict]:
        bm = await init_async(job="batch-list")

        effective_save_dir = save_dir or bm.session_info.save_dir
        if not effective_save_dir:
            raise RuntimeError("No save_dir configured. Use --save-dir to specify.")

        # cloudpathlib.AnyPath is a factory returning a concrete CloudPath or
        # pathlib.Path; its __new__ is type-ignored upstream so mypy cannot see
        # the real return type. Cast to the documented runtime union.
        save_root = cast("CloudPath | Path", AnyPath(effective_save_dir))
        batch_dir: CloudPath | Path = save_root / "batch"

        if not batch_dir.exists():
            return []

        jobs = []
        # List job directories
        job_dirs = cast("list[CloudPath | Path]", sorted(batch_dir.iterdir(), reverse=True)[:limit])

        for job_dir in job_dirs:
            if not job_dir.is_dir():
                continue

            manifest_path = job_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = BatchJobManifest.model_validate_json(manifest_path.read_text())
                jobs.append(
                    {
                        "job_id": manifest.job_id,
                        "model": manifest.model,
                        "submitted_at": manifest.submitted_at,
                        "request_count": manifest.request_count,
                        "vertex_job_name": manifest.vertex_job_name,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse manifest {manifest_path}: {e}")
                continue

        return jobs

    try:
        jobs = asyncio.run(run_list())

        if json_output:
            click.echo(json.dumps(jobs, indent=2))
        elif not jobs:
            click.echo("No batch jobs found.")
            click.echo("Jobs are stored in: <save_dir>/batch/<job_id>/")
        else:
            click.echo(f"Recent batch jobs ({len(jobs)} found):\n")
            for job in jobs:
                click.echo(f"  {job['job_id']}")
                click.echo(f"    Model: {job['model']}")
                click.echo(f"    Submitted: {job['submitted_at']}")
                click.echo(f"    Requests: {job['request_count']}")
                click.echo()

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        if json_output:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    batch()
