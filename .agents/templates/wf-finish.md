---
alias:
  - wf-finish
description: buttermilk finish template -- defines task delivery via PR targeting dev, the merge queue gate, and independent QA review of functional changes before the owner merges.
id: wf-finish
tags:
  - wf-template
  - finish
  - buttermilk
title: buttermilk Task Finish Template
type: template
---

## What this template does

Specifies how tasks in `qut-dmrc/buttermilk` finish. Every change delivers as a PR against `dev`. Functional changes require independent QA review on the PR. The repository owner approves and merges through the `dev` merge queue.

## Finish Policy

- **Target branch**: `dev`. Never target `stable`.
- **Delivery mechanism**: Push a feature branch and open a pull request against `dev` using `.github/pull_request_template.md` (`gh pr create --base dev`). Never push directly to `dev`.
- **Merging**: Workers and QA never merge. `dev` requires two approving reviews, the `Lint` and `review-attestation` checks, and a squash merge through the merge queue. The repository owner approves and enqueues the PR.
- **Commit trailer**: Commits must carry `Task: <task-id>` (and `Epic: <epic-id>` if applicable). Repeat the `Task:` line in the PR body.
- **QA review**:
  - **Required**: Any change to code under `buttermilk/`, tests, Hydra config under `buttermilk/conf/`, pipelines, flows, storage schemas, dependencies (`pyproject.toml`, `uv.lock`), containers, or `.github/workflows/`. `/dispatch` mints a follow-up verifying task.
  - **None**: Documentation-only changes (markdown, docstrings, comments) with no functional impact.

## Worker Completion Checklist

Before marking `status: done`, the worker must:

1. Sync the environment: `uv sync --extra dev`.
2. Run the unit suite and confirm it passes:
   `uv run pytest tests/ --ignore=tests/endtoend -m "not (integration or endtoend or slow or demo)"`.
3. Run lint and format checks and confirm they pass: `uv run ruff check` and `uv run ruff format --check`.
4. Run `uv run pre-commit run --all-files` and confirm it passes.
5. Start any new `.py` file with `# SPDX-License-Identifier: GPL-3.0-or-later`.
6. Push the feature branch (`task/<id>-<slug>`) and open a PR against `dev` with the PR template filled in.
7. Confirm CI has started on the PR (`gh pr checks <pr-number>`); record any failure verbatim on the task.
8. Check off each acceptance criterion on the PKB task with pinpoint evidence (`file:line`, command output, PR link).
9. Mark the task `status: done` and release the claim.

## Follow-up QA Task Specification

Where QA is required, the verifying task runs independently on a clean checkout of the PR branch:

- **Title**: `QA: <primary task title>`
- **Parent**: Same parent as primary task
- **Depends on**: `[<primary-task-id>]`
- **Workflow**: Composes independent verification workflow
- **Goal**: Independently verify the PR on a clean checkout against the literal acceptance criteria: run the checklist test and lint commands, confirm new tests exercise the real code path with no mocks of `buttermilk.*` code, and confirm CI is green. Post the verdict with evidence as a PR comment and on the task. Do not merge; merging is the repository owner's.
