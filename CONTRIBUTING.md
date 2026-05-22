# Contributing to buttermilk

Buttermilk is research software for the humanities and social sciences. Contributions are welcome — including bug reports, design discussions, and code.

## Before you start

- **Discuss first, code second.** For non-trivial changes, open an issue describing the problem and your proposed approach before sending a PR. Research priorities and methodological choices belong on the design surface, not in code review.
- **Read the [Code of Conduct](CODE_OF_CONDUCT.md).** Participation in this project is conditional on agreeing to it.

## Development setup

Buttermilk uses [uv](https://github.com/astral-sh/uv) for dependency and environment management.

```bash
git clone https://github.com/qut-dmrc/buttermilk.git
cd buttermilk
uv sync --extra dev --upgrade
uv run pre-commit install
```

To run the tests (excluding slow / cloud-dependent tests):

```bash
uv run pytest -m "not slow and not endtoend and not demo"
```

Test markers are declared in `pyproject.toml`:

- `integration` — multi-component tests, mocked at the edges.
- `endtoend` — full environment, real APIs and storage.
- `demo` — slow, expensive, human-verified.
- `slow` — anything over ~10s.
- `scheduled` — runs on cron, not in PR CI.

## Pull request process

1. Branch from `dev` (or the current default development branch).
2. Make your change. Keep PRs focused — one concern per PR.
3. Ensure `uv run pre-commit run --all-files` passes locally.
4. Open a PR using the [template](.github/pull_request_template.md).
5. CI will run linting, tests, and an automated PR review pipeline. Address feedback.
6. Once approved, a maintainer will merge.

## Coding conventions

- **No mocks of `buttermilk.*` code or business logic.** Mock only at system boundaries — network, filesystem, time, environment variables. See `.agent/workflows/TESTING.md`.
- **Fail fast, fail loud.** Do not add defensive code around tracing, logging, or data-saving failures. Observability is load-bearing for research integrity.
- **Methodology belongs to the researcher.** LLM generation parameters (`temperature`, `max_tokens`, `top_p`, `top_k`), model selection, and prompt templates must be explicitly configured — no library defaults.

## Licensing

By submitting a contribution to buttermilk, you agree to license it under **GPL-3.0-or-later**, matching the project license. No DCO sign-off is required, and there is no separate CLA.

### SPDX headers on new files

New `.py` files must begin with:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
```

Existing files are not modified retroactively — backfill is tracked as a separate issue.

### License FAQ

**If I fork buttermilk and run it as a hosted service, do I have to share my changes?**
No. GPL-3.0 triggers copyleft obligations only on *distribution* of the software, not on network or SaaS use. If you offer a hosted service built on buttermilk, you are not obliged to share your modifications. (This is the key difference between GPL and AGPL; buttermilk chose plain GPL deliberately.) You are obliged to share modifications if you distribute the modified package itself (e.g. on PyPI, in a Docker image you publish, or as a binary).

**Can I depend on buttermilk from a permissive (MIT / Apache-2.0) project?**
Depending on buttermilk at runtime does not require relicensing your project. However, redistributing buttermilk as part of a single combined work (e.g. a bundled wheel, a Docker image you publish) does. If in doubt, ask before publishing.

**Can I include buttermilk in commercial / proprietary research tools?**
Yes, with the same rules: internal use is unrestricted; distribution must include source for any modifications to buttermilk itself.

## Release process

Buttermilk uses `setuptools_scm` + git tags for versioning — there is no version string to bump in source.

Release flow (maintainers only):

1. Ensure `CHANGELOG.md` has notes under `[Unreleased]`.
2. Update `CITATION.cff` `version:` field to the upcoming tag.
3. Move `[Unreleased]` notes under a new dated heading in `CHANGELOG.md`.
4. Tag the release: `git tag v0.0.N -m "..." && git push --tags`.
5. Run `gh release create v0.0.N --notes-from-file ...` (or via the web UI).
6. The `publish.yml` workflow fires on `release: published`, builds the wheel + sdist, and publishes to PyPI via **trusted publishing (OIDC)** — no API token required.

### One-time PyPI trusted-publisher setup

A repository admin must, once:

1. Create the buttermilk project on PyPI (first publish needs an alternate path — see PyPI docs).
2. Go to **Manage → Publishing → Add a new publisher** on PyPI.
3. Add a GitHub trusted publisher with:
   - Owner: `qut-dmrc`
   - Repository: `buttermilk`
   - Workflow: `publish.yml`
   - Environment: `pypi` (recommended — restrict to protected releases).

See [PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/) for details.

## Reporting issues

Use the GitHub issue templates:

- **Bug report** — something is broken.
- **Feature request** — something is missing.
- **Question** — usage / design questions and discussion.

For security issues, see [SECURITY.md](SECURITY.md) — do not file public issues.
