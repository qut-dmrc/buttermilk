#!/usr/bin/env python3
"""Registry cutover for the ClientType deletion — APPLY AT MERGE TIME.

This is the deferred, destructive half of the "delete the ClientType enum and route
every model through a full provider-prefixed litellm name" change (branch
feat/delete-client-type, follow-up to PR #438).

The CODE change removes `ClientType` and the model-routing/auth mapping machinery
from buttermilk/_core/llms.py. `LLMConfig` no longer has a `client_type` field and
now REQUIRES a full provider-prefixed `litellm_model` (e.g. "vertex_ai/gemini-3-...",
"azure/gpt-5-mini", "azure_ai/grok-4-1-fast-non-reasoning"). Therefore the live GCP
secret `dev__llm__connections` (the SSoT; ~/.cache/buttermilk/models/models.json is
just a cache of it) MUST be rewritten in lockstep:
  * pre-merge: rewriting the registry breaks live runs still on the old code, and
  * post-merge: leaving entries without a `litellm_model` makes LLMConfig validation
    raise for every affected model.

For each entry this script:
  * derives a full `litellm_model` from the old `client_type` + `configs.model`
  * lifts `region`  (from configs.region) and `api_version` (from configs.api_version)
    to top-level LLMConfig fields
  * lifts `base_url` to top-level if it was only under configs
  * drops the now-unused `client_type` field
  * leaves `model_info`, `parameters`, `configs.model` untouched

KNOWN-BROKEN (intentional, per Nic 2026-06-22 "break grok support for now"):
  The 2 Grok-on-Vertex reasoning models (xai/grok-4.20, xai/grok-4-1-fast-reasoning)
  used the deleted VERTEX_XAI OpenAI-compat-on-Vertex shim. litellm 1.83.0 has NO
  working native vertex_ai/xai/ path and there is no direct xAI key, so these are
  mapped to a non-functional `vertex_ai/<name>` and WILL FAIL at call time. They are
  kept in the registry (validate fine) but are non-functional until redeployed on
  Azure AI (mirror the working azure_ai/grok-4-1-fast-non-reasoning entry).

NOTE: google/gemini-3-pro-preview is 404 in prosocial-443205 on every path
(pre-existing stale id, not a regression); migrated for consistency.

Usage:
    uv run python scripts/migrate_to_litellm_model.py            # dry-run (default)
    uv run python scripts/migrate_to_litellm_model.py --apply    # writes a new secret version

Requires GCP creds with secretmanager.versions.access + .add on the project.
"""

from __future__ import annotations

import argparse
import json
import sys

PROJECT = "prosocial-443205"
SECRET_NAME = "dev__llm__connections"

GEMINI_REGION = "global"
GROK_BROKEN_REGION = "us-east5"

# Grok-on-Vertex reasoning ids -> corrected publisher ids (kept for record; still broken).
GROK_REASONING_IDS = {
    "xai/grok-4.20": "xai/grok-4.20-reasoning",
    "xai/grok-4-1-fast-reasoning": "xai/grok-4.1-fast-reasoning",
}


def _strip(prefix: str, name: str) -> str:
    return name[len(prefix) :] if name.startswith(prefix) else name


def derive_litellm_model(name: str, entry: dict) -> tuple[str, str | None, str | None]:
    """Return (litellm_model, region, note) for an entry.

    `note` is a human-readable warning string or None.
    """
    ct = entry.get("client_type")
    cfg = entry.get("configs", {}) or {}
    model = cfg.get("model") or name
    region = cfg.get("region")

    # Already-native entries that carry an explicit litellm_model (e.g. the working
    # azure_ai grok-non-reasoning): keep it verbatim.
    existing = entry.get("litellm_model")
    if existing:
        return existing, region, None

    if ct == "azure":
        return f"azure/{model}", None, None
    if ct == "openai":
        # Direct OpenAI (no explicit litellm_model handled above) -> openai/<model>.
        return f"openai/{model}", None, None
    if ct == "anthropic":
        return model, region, None  # litellm uses bare anthropic ids
    if ct == "anthropic_vertex":
        return f"vertex_ai/{model}", region, None
    if ct == "gemini":
        return model, region, None  # AI-Studio gemini API uses bare ids
    if ct in ("gemini_vertex", "vertex_openai") and model.startswith("google/"):
        # Gemini on Vertex (native): strip google/, force global region.
        return f"vertex_ai/{_strip('google/', model)}", GEMINI_REGION, None
    if ct == "llama_vertex":
        return f"vertex_ai/{model}", region, None  # keep meta/
    if ct == "deepseek_vertex":
        return f"vertex_ai/{model}", region, None  # keep deepseek-ai/
    if ct == "mistral_vertex":
        return f"vertex_ai/{_strip('mistralai/', model)}", region, None
    if ct == "huggingface":
        return f"huggingface/{model}", region, None
    if ct == "zentropi":
        return f"zentropi/{model}", region, None
    if ct == "vertex_openai" and model.startswith("xai/"):
        # Grok-on-Vertex reasoning: VERTEX_XAI shim deleted; no working native path.
        fixed = GROK_REASONING_IDS.get(name, model)
        return (
            f"vertex_ai/{fixed}",
            GROK_BROKEN_REGION,
            "BROKEN: Grok-on-Vertex reasoning has no working litellm route after VERTEX_XAI deletion; non-functional until redeployed on Azure AI.",
        )

    # Unknown / unmapped — leave a best-effort prefix and warn.
    return model, region, f"UNMAPPED client_type {ct!r}; left litellm_model={model!r} verbatim"


def migrate(registry: dict) -> tuple[dict, list[str]]:
    """Return (new_registry, change_log). Pure function — no I/O."""
    out = json.loads(json.dumps(registry))  # deep copy
    log: list[str] = []

    for name, entry in out.items():
        if not isinstance(entry, dict):
            log.append(f"SKIP  {name}: not an object")
            continue
        litellm_model, region, note = derive_litellm_model(name, entry)
        cfg = entry.get("configs", {}) or {}

        entry["litellm_model"] = litellm_model
        if region:
            entry["region"] = region
        # Lift api_version to top-level (Azure batch SDK path).
        api_version = entry.get("api_version") or cfg.get("api_version")
        if api_version:
            entry["api_version"] = api_version
        # Lift base_url to top-level if only present under configs.
        if not entry.get("base_url") and cfg.get("base_url"):
            entry["base_url"] = cfg["base_url"]
        # Gemini native path constructs its own endpoint — drop the old openapi base_url.
        if litellm_model.startswith("vertex_ai/gemini"):
            entry.pop("base_url", None)
        entry.pop("client_type", None)

        msg = f"{name}: litellm_model -> {litellm_model!r}"
        if region:
            msg += f", region={region}"
        if note:
            msg += f"  [{note}]"
        log.append(msg)

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

    print("=== Cutover plan (litellm_model + region; drop client_type) ===")
    for line in log:
        print("  ", line)

    if new_registry == registry:
        print("\nNo changes needed. Nothing to write.")
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
