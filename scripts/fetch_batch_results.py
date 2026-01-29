#!/usr/bin/env python3
"""Script to fetch results for a specific batch job.

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

from buttermilk import bm, logger
from buttermilk._core.config_bootstrap import init_async
from buttermilk._core.vertex_batch import BatchJobManager, BatchJobManifest

# Configure logging
logging.basicConfig(level=logging.INFO)


async def fetch_results(job_id: str):
    """Fetch results for a batch job."""
    # 1. Initialize Buttermilk
    logger.info("Initializing buttermilk context...")
    # We use a generic job name, assuming session config handles project/location
    await init_async(project_name="batch_ops", job="fetch_results")

    # 2. Locate Manifest
    logger.info(f"Looking for manifest for job {job_id}...")
    manifest_path = None

    # Strategy 1: Check stable persistent path (O(1) lookup)
    if bm.session_info.save_dir_base:
        try:
            from cloudpathlib import AnyPath

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
        # heuristic: try to find based on runs dir from save_dir
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

    if not manifest_path:
        logger.error("Could not locate manifest in any location.")
        sys.exit(1)

    try:
        manifest_data = json.loads(manifest_path.read_text())
        manifest = BatchJobManifest(**manifest_data)
        logger.info(f"Loaded manifest for job {job_id} (Vertex Job: {manifest.vertex_job_name})")
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        sys.exit(1)

    # 3. Check Status
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

    # Initialize manager with specific location if possible, otherwise use default
    # Note: buttermilk's bm.genai client is pre-configured. We might need to create a new client
    # or rely on the underlying library to handle global vs regional if configured correctly.

    # For now, let's try to infer if we need a specific location client.
    # If the job is global, we should use a global client or us-central1 often works.
    # But the error 404 suggests mismatched location.

    from google.genai import Client

    # If we need to override the location, we might need a fresh client.
    # Buttermilk doesn't easily expose client factory with override, so we'll try to use the raw client
    # if the default one fails, or just try to pass the full name which SHOULD work if the client is
    # global-aware.

    # Let's try to create a specific client for this location if it differs from default
    client = bm.genai
    if job_location != "us-central1":  # Assuming default is us-central1
        try:
            # Attempt to create a fresh client for the specific location
            # This requires getting credentials/project from existing config
            import os

            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                # Fallback to session info project name if env var not consistent
                # (though config_bootstrap sets env var)
                project = bm.session_info.project_name

            logger.info(f"Re-initializing client for location: {job_location} (Project: {project})")
            client = Client(vertexai=True, project=project, location=job_location)
        except Exception as e:
            logger.warning(f"Failed to create location-specific client: {e}. Using default.")
            client = bm.genai

    manager = BatchJobManager(client=client)

    try:
        # Get the actual Vertex batch job object
        # Note: client.batches.get() might need just the ID if the client is location-aware,
        # or the full name. The error said "BatchPredictionJob does not exist", which strongly implies
        # looking in the wrong region.

        # If the name is full resource name, the client *should* handle it, but sometimes it strips
        # project/location project and uses its internal config.
        # Let's try to pass just the ID if the full name fails, or vice versa.

        job_resource_name = manifest.vertex_job_name

        logger.info(f"Polling status for: {job_resource_name}")
        vertex_job = manager.client.batches.get(name=job_resource_name)
        logger.info(f"Current Job State: {vertex_job.state}")

        # Wait for completion (this handles polling if it's still running)
        logger.info("Waiting for completion...")
        completed_job = await manager.wait_for_completion(vertex_job)

    except Exception as e:
        logger.error(f"Error checking/waiting for job: {e}")
        sys.exit(1)

    # 4. Parse Results
    logger.info(f"Job completed. Fetching results from {manifest.output_uri}...")
    try:
        results = manager.parse_results(manifest.output_uri, manifest.requests)

        # Summary
        success_count = sum(1 for r in results if not r.error)
        error_count = len(results) - success_count

        logger.info("=" * 40)
        logger.info("Batch Processing Summary")
        logger.info(f"Total Requests: {len(manifest.requests)}")
        logger.info(f"Results Parsed: {len(results)}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("=" * 40)

        # Optional: Save results to a local file for inspection
        output_file = f"batch_results_{job_id}.jsonl"
        with open(output_file, "w") as f:
            for res in results:
                f.write(res.model_dump_json() + "\n")
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to parse results: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fetch Vertex Batch Results")
    parser.add_argument("job_id", help="The internal batch job ID (e.g., batch_xyz)")
    args = parser.parse_args()

    asyncio.run(fetch_results(args.job_id))


if __name__ == "__main__":
    main()
