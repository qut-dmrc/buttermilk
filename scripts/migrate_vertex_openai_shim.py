#!/usr/bin/env python3
"""Registry cutover for the vertex_openai shim removal — APPLY AT MERGE TIME.

This is the deferred, destructive half of the "remove the legacy vertex_openai
OpenAI-compat shim" change. The CODE change (branch feat/remove-vertex-openai-shim)
deletes the `vertex_openai` ClientType, so the live registry must be updated to the
new client types AT THE SAME TIME the code merges — otherwise:
  * pre-merge: flipping the registry breaks live runs still on the old code, and
  * post-merge: leaving the registry on `vertex_openai` makes LLMs() construction
    raise (unknown client_type) for all 7 affected models.

It rewrites the GCP secret `dev__llm__connections` (the SSoT; the local
~/.cache/buttermilk/models/models.json is just a cache of it) for exactly the 7
models that used `client_type: vertex_openai`:

  Gemini (5) -> client_type: gemini_vertex   (litellm native vertex_ai/gemini-...)
     * drop base_url (the OpenAI-compat openapi endpoint)
     * configs.project_id = <project>, configs.region = "global"
       (gemini-3.x models only resolve under the `global` Vertex location)

  Grok-on-Vertex (2) -> client_type: vertex_xai  (Vertex OpenAI-compat endpoint,
     the ONLY working route: litellm 1.83.0 has no native vertex_ai/xai/ path and
     there is no direct xAI API key)
     * keep base_url, but pin location to us-east5 (where all 4 grok ids resolve;
       global only serves the grok-4.1-fast-* family)
     * configs.project_id = <project>, configs.region = "us-east5"
     * FIX the model id to the real Vertex publisher id:
         xai/grok-4-1-fast-reasoning -> xai/grok-4.1-fast-reasoning
         xai/grok-4.20               -> xai/grok-4.20-reasoning
       (the old ids 404 even on the current shim.)

NOTE on google/gemini-3-pro-preview: this model id is currently 404 in project
prosocial-443205 on BOTH the old shim and the native path (a pre-existing stale
registry entry, not a migration regression). This script migrates its client_type
for consistency but it will remain non-functional until the id is corrected or the
entry removed. Review separately.

Usage:
    # dry-run (default) — prints the diff, writes nothing:
    uv run python scripts/migrate_vertex_openai_shim.py

    # actually add a new secret version:
    uv run python scripts/migrate_vertex_openai_shim.py --apply

Requires GCP creds with secretmanager.versions.access + .add on the project.
"""

from __future__ import annotations

import argparse
import json
import sys

PROJECT = "prosocial-443205"
SECRET_NAME = "dev__llm__connections"

GEMINI_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash",
]

# display-name -> corrected Vertex publisher model id
GROK_MODELS = {
    "xai/grok-4-1-fast-reasoning": "xai/grok-4.1-fast-reasoning",
    "xai/grok-4.20": "xai/grok-4.20-reasoning",
}

GROK_REGION = "us-east5"
GEMINI_REGION = "global"


def migrate(registry: dict) -> tuple[dict, list[str]]:
    """Return (new_registry, change_log). Pure function — no I/O."""
    out = json.loads(json.dumps(registry))  # deep copy
    log: list[str] = []

    for name in GEMINI_MODELS:
        entry = out.get(name)
        if entry is None:
            log.append(f"SKIP  {name}: not present in registry")
            continue
        if entry.get("client_type") != "vertex_openai":
            log.append(f"SKIP  {name}: client_type is {entry.get('client_type')!r}, not vertex_openai")
            continue
        entry["client_type"] = "gemini_vertex"
        entry.pop("base_url", None)  # native path constructs its own endpoint
        cfg = entry.setdefault("configs", {})
        cfg["project_id"] = PROJECT
        cfg["region"] = GEMINI_REGION
        log.append(f"GEMINI {name}: client_type -> gemini_vertex, drop base_url, configs.region=global, configs.project_id={PROJECT}")

    for name, real_id in GROK_MODELS.items():
        entry = out.get(name)
        if entry is None:
            log.append(f"SKIP  {name}: not present in registry")
            continue
        if entry.get("client_type") != "vertex_openai":
            log.append(f"SKIP  {name}: client_type is {entry.get('client_type')!r}, not vertex_openai")
            continue
        entry["client_type"] = "vertex_xai"
        # repoint base_url location to us-east5 (where all grok ids resolve)
        entry["base_url"] = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{GROK_REGION}/endpoints/openapi"
        cfg = entry.setdefault("configs", {})
        old_id = cfg.get("model")
        cfg["model"] = real_id
        cfg["project_id"] = PROJECT
        cfg["region"] = GROK_REGION
        log.append(
            f"GROK   {name}: client_type -> vertex_xai, configs.model {old_id!r} -> {real_id!r}, "
            f"base_url location -> {GROK_REGION}, configs.region={GROK_REGION}"
        )

    return out, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Write a new secret version (default: dry-run)")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--secret", default=SECRET_NAME)
    args = ap.parse_args()

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    latest = f"projects/{args.project}/secrets/{args.secret}/versions/latest"
    raw = client.access_secret_version(request={"name": latest}).payload.data.decode("UTF-8")
    registry = json.loads(raw)

    new_registry, log = migrate(registry)

    print("=== Cutover plan ===")
    for line in log:
        print(" ", line)

    if new_registry == registry:
        print("\nNo changes needed (already migrated or no matching entries). Nothing to write.")
        return 0

    if not args.apply:
        print("\nDRY-RUN: no secret version written. Re-run with --apply to add a new version.")
        return 0

    payload = json.dumps(new_registry, indent=2).encode("UTF-8")
    parent = f"projects/{args.project}/secrets/{args.secret}"
    resp = client.add_secret_version(request={"parent": parent, "payload": {"data": payload}})
    print(f"\nAPPLIED: added new secret version {resp.name}")
    print("Clear the local cache so it re-fetches: rm -f ~/.cache/buttermilk/models/models.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
