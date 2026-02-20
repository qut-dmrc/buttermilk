#!/usr/bin/env python3
"""Script to fetch results for a specific batch job.

Supports both Vertex AI and OpenAI/Azure batch jobs. The manifest type
is auto-detected: if it contains `openai_batch_id` it's routed to the
OpenAI path, otherwise to the Vertex path.

Usage:
    python3 scripts/fetch_batch_results.py <job_id>

Example:
    python3 scripts/fetch_batch_results.py batch_6104d445479a
"""

import argparse
import asyncio
import json
import logging
import sys

from cloudpathlib import AnyPath

from buttermilk import bm, logger
from buttermilk._core.config_bootstrap import init_async

# Configure logging
logging.basicConfig(level=logging.INFO)


def _locate_manifest(job_id: str) -> AnyPath | None:
    """Locate manifest file across standard storage locations.

    Tries three strategies in order:
    1. Stable persistent path at {save_dir_base}/{project}/_batches/{job_id}/
    2. Current session path at {save_dir}/batch/{job_id}/
    3. Deep search across all runs in the bucket
    """
    manifest_path = None

    # Strategy 1: Check stable persistent path (O(1) lookup)
    if bm.session_info.save_dir_base:
        try:
            base = AnyPath(bm.session_info.save_dir_base)
            stable_path = base / bm.session_info.project_name / "_batches" / job_id / "manifest.json"
            if stable_path.exists():
                manifest_path = stable_path
                logger.info(f"Found manifest at stable location: {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to check stable path: {e}")

    # Strategy 2: Check current session (Legacy/Fallback)
    if not manifest_path:
        save_dir = bm.session_info.save_dir
        if save_dir:
            try:
                session_path = AnyPath(f"{save_dir}/batch/{job_id}/manifest.json")
                if session_path.exists():
                    manifest_path = session_path
                    logger.info(f"Found manifest in current session: {manifest_path}")
            except Exception:
                pass

    # Strategy 3: Search all runs (Deep Search Fallback)
    if not manifest_path and bm.session_info.save_dir:
        try:
            save_dir = bm.session_info.save_dir
            parts = save_dir.split("/")
            if len(parts) > 4 and "gs:" in parts[0]:
                bucket = parts[2]
                runs_root = f"gs://{bucket}/runs"
                logger.info(f"Searching for manifest in {runs_root}...")
                runs_path = AnyPath(runs_root)
                found_manifests = list(runs_path.glob(f"**/batch/{job_id}/manifest.json"))
                if found_manifests:
                    manifest_path = found_manifests[0]
                    logger.info(f"Found manifest at alternate location: {manifest_path}")
        except Exception as e:
            logger.warning(f"Deep search failed: {e}")

    return manifest_path


async def _fetch_vertex_results(job_id: str, manifest_data: dict) -> None:
    """Fetch results for a Vertex AI batch job."""
    from buttermilk._core.vertex_batch import BatchJobManager, BatchJobManifest

    manifest = BatchJobManifest(**manifest_data)
    logger.info(f"Loaded Vertex manifest for job {job_id} (Vertex Job: {manifest.vertex_job_name})")

    # Parse location from vertex_job_name
    # Format: projects/{project}/locations/{location}/batchPredictionJobs/{id}
    job_location = "us-central1"  # Default
    try:
        parts = manifest.vertex_job_name.split("/")
        if "locations" in parts:
            loc_idx = parts.index("locations") + 1
            if loc_idx < len(parts):
                job_location = parts[loc_idx]
                logger.info(f"Detected job location from manifest: {job_location}")
    except Exception:
        logger.warning(f"Could not parse location from job name: {manifest.vertex_job_name}")

    from google.genai import Client

    client = bm.genai
    if job_location != "us-central1":
        try:
            import os

            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                project = bm.session_info.project_name

            logger.info(f"Re-initializing client for location: {job_location} (Project: {project})")
            client = Client(vertexai=True, project=project, location=job_location)
        except Exception as e:
            logger.warning(f"Failed to create location-specific client: {e}. Using default.")
            client = bm.genai

    manager = BatchJobManager(client=client)

    try:
        job_resource_name = manifest.vertex_job_name
        logger.info(f"Polling status for: {job_resource_name}")
        vertex_job = manager.client.batches.get(name=job_resource_name)
        logger.info(f"Current Job State: {vertex_job.state}")

        logger.info("Waiting for completion...")
        await manager.wait_for_completion(vertex_job)
    except Exception as e:
        logger.error(f"Error checking/waiting for job: {e}")
        raise

    # Parse Results
    logger.info(f"Job completed. Fetching results from {manifest.output_uri}...")
    try:
        results = manager.parse_results(manifest.output_uri, manifest.requests)
        _print_summary_and_save(job_id, results, len(manifest.requests))
    except Exception as e:
        logger.error(f"Failed to parse results: {e}")
        raise


def _fetch_openai_results(job_id: str, manifest_data: dict) -> None:
    """Fetch results for an OpenAI/Azure batch job."""
    from buttermilk._core.vertex_batch import OpenAIBatchJobManager, OpenAIBatchManifest
    from buttermilk.batch.executors.openai import _create_openai_batch_client

    manifest = OpenAIBatchManifest(**manifest_data)
    logger.info(f"Loaded OpenAI manifest for job {job_id} (OpenAI Batch ID: {manifest.openai_batch_id}, Model: {manifest.model})")

    # Create client from the model registry
    try:
        client, endpoint = _create_openai_batch_client(manifest.model)
    except ValueError:
        # Model may not be in current config; try to use a generic OpenAI client
        logger.warning(f"Model '{manifest.model}' not in current config. Attempting direct OpenAI client.")
        from openai import OpenAI

        client = OpenAI()
        endpoint = "/v1/chat/completions"

    manager = OpenAIBatchJobManager(
        client=client,
        endpoint=endpoint,
    )

    result = manager.fetch_results(job_id)

    if "error" in result:
        logger.error(f"Fetch failed: {result['error']}")
        raise RuntimeError(result["error"])
    elif "summary" in result and "results" in result:
        summary = result["summary"]
        results_data = result["results"]

        logger.info("=" * 40)
        logger.info("Batch Processing Summary (OpenAI)")
        logger.info(f"  Job ID: {summary['job_id']}")
        logger.info(f"  Model: {summary['model']}")
        logger.info(f"  Requests: {summary['request_count']}")
        logger.info(f"  Success: {summary['success_count']}")
        logger.info(f"  Errors: {summary['error_count']}")
        if summary.get("total_cost_usd") is not None:
            logger.info(f"  Estimated Cost: ${summary['total_cost_usd']:.4f}")
        logger.info("=" * 40)

        output_file = f"batch_results_{job_id}.jsonl"
        with open(output_file, "w") as f:
            for r in results_data:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Results saved to {output_file}")
    else:
        # Still running
        status = result.get("status", "unknown")
        logger.info(f"Job {job_id} status: {status}")
        if result.get("completed") is not None:
            logger.info(f"  Progress: {result.get('completed', 0)}/{result.get('total', '?')}")
        logger.info("Job is still running. Check back later.")


def _print_summary_and_save(job_id: str, results: list, request_count: int) -> None:
    """Print summary and save results to JSONL file."""
    success_count = sum(1 for r in results if not r.error)
    error_count = len(results) - success_count

    logger.info("=" * 40)
    logger.info("Batch Processing Summary (Vertex)")
    logger.info(f"Total Requests: {request_count}")
    logger.info(f"Results Parsed: {len(results)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("=" * 40)

    output_file = f"batch_results_{job_id}.jsonl"
    with open(output_file, "w") as f:
        for res in results:
            f.write(res.model_dump_json() + "\n")
    logger.info(f"Results saved to {output_file}")


async def fetch_results(job_id: str):
    """Fetch results for a batch job (auto-detects Vertex vs OpenAI)."""
    # 1. Initialize Buttermilk
    logger.info("Initializing buttermilk context...")
    await init_async(project_name="batch_ops", job="fetch_results")

    # 2. Locate Manifest
    logger.info(f"Looking for manifest for job {job_id}...")
    manifest_path = _locate_manifest(job_id)

    if not manifest_path:
        logger.error("Could not locate manifest in any location.")
        sys.exit(1)

    try:
        manifest_data = json.loads(manifest_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        sys.exit(1)

    # 3. Detect manifest type and route
    try:
        if "openai_batch_id" in manifest_data:
            logger.info("Detected OpenAI/Azure batch manifest.")
            _fetch_openai_results(job_id, manifest_data)
        else:
            logger.info("Detected Vertex AI batch manifest.")
            await _fetch_vertex_results(job_id, manifest_data)
    finally:
        # Ensure all logs and traces are flushed before exiting, even on error
        await bm.graceful_shutdown()


def main():
    parser = argparse.ArgumentParser(description="Fetch Batch Results (Vertex AI or OpenAI)")
    parser.add_argument("job_id", help="The internal batch job ID (e.g., batch_xyz)")
    args = parser.parse_args()

    asyncio.run(fetch_results(args.job_id))


if __name__ == "__main__":
    main()
