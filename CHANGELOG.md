# Changelog

All notable changes to buttermilk are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) via `setuptools_scm` and git tags.

Earlier history (before v0.0.1) was developed under a different versioning scheme; see git history for full provenance.

## [Unreleased]

### Added
- LICENSE (GPL-3.0-or-later) and `license` metadata in `pyproject.toml`.
- PyPI classifiers, keywords, and `[project.urls]` for discoverability.
- `py.typed` marker (PEP 561) so consumers' type checkers honour buttermilk's type hints.
- `endtoend` pytest marker declaration (previously emitted `PytestUnknownMarkWarning`).
- `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- GitHub issue templates and `.github/workflows/publish.yml` for PyPI trusted publishing.
- `LICENSES.md` — dependency license audit confirming GPL-3.0 compatibility.

### Changed
- Upper-bounded previously unbounded core deps: `numpy`, `chromadb`, `langchain`, `regex`, `keyring`.
- `README.md` rewritten with quickstart, badges, accurate installation instructions, and no dead doc links.

### Fixed
- `buttermilk/__init__.py`: removed stale `initialize_session_bm` entry from `__all__` — `from buttermilk import *` previously raised `AttributeError`.

### Removed
- In-tree backup files (`*.bak`, `*.backup`) committed before the `.gitignore` rule.
- Conflicting `ruff.toml` (consolidated into `pyproject.toml` `[tool.ruff]`).

## [0.0.27] - 2026
- Per-message hashing for experiment identity (#378).

## [0.0.26] - 2026
- ChromaDBEmbeddings `read_only=True` skips embedding function — breaks queries (#382).

## [0.0.25] - 2026
- CI: unblock PR Checks — move `continue-on-error` into gatekeeper step (#385).

## [0.0.24] - 2026
- CI: grant caller-level permissions to merge-prep shims (#384).

## [0.0.23] - 2026
- CI: install aops merge-prep / PR pipeline (#383).

## [0.0.22] - 2026
- Move design/decisions/reference docs to brain PKB.

## [0.0.21] - 2026
- Audit: remove absolute paths from `CORE.md` (#381).

## [0.0.20] - 2026
- Add `.dockerignore`.

## [0.0.19] - 2026
- Zotero updater.

## [0.0.18] - 2026
- Processor improvements.

## [0.0.17] - 2026
- Install enforcer agent replacing custodiet (#380).

## [0.0.16] - 2026
- Move `seaborn` from dev extras to core dependencies (#379).

## [0.0.15] - 2026
- Add missing `slow`/`integration` markers to all end-to-end test files (#377).

## [0.0.14] - 2026
- Resolve flaky tests and CI errors blocking merge queue (#375).

## [0.0.13] - 2026
- Merge non-config `variant_params` into `LLMProcessor` template vars (#374).

## [0.0.12] - 2026
- JMESPath inputs mapping for per-record processor config overrides (#364).

## [0.0.11] - 2026
- Configurable error handling for pipeline variant expansion (#371, #372).

## [0.0.10] - 2026
- Fix 24 unit test failures from CI run #367 (#369).

## [0.0.9] - 2026
- Mark slow tests (>10s) with `@pytest.mark.slow` (#365).

## [0.0.8] - 2026
- Mistral updates.

## [0.0.7] - 2026
- VSCode updates.

## [0.0.6] - 2026
- Add deepseek-v3 to structured output tests and fix test issues (#368).

## [0.0.5] - 2026
- Fix `LLMProcessor` `variant_params` not merged into template vars (#367).

## [0.0.4] - 2026
- Model updates.

## [0.0.3] - 2026
- Fix Vertex batch schema rejection of integer `Literal` enums for Gemini models (#358).

## [0.0.2] - 2026
- Refactor: use `_is_claude_model()` helper; add OpenAI batch processor tests (#356).

## [0.0.1] - 2026
- Modernise release versioning with `setuptools_scm` and CI (#354).

[Unreleased]: https://github.com/qut-dmrc/buttermilk/compare/v0.0.27...HEAD
[0.0.27]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.27
[0.0.26]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.26
[0.0.25]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.25
[0.0.24]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.24
[0.0.23]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.23
[0.0.22]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.22
[0.0.21]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.21
[0.0.20]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.20
[0.0.19]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.19
[0.0.18]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.18
[0.0.17]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.17
[0.0.16]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.16
[0.0.15]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.15
[0.0.14]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.14
[0.0.13]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.13
[0.0.12]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.12
[0.0.11]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.11
[0.0.10]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.10
[0.0.9]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.9
[0.0.8]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.8
[0.0.7]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.7
[0.0.6]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.6
[0.0.5]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.5
[0.0.4]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.4
[0.0.3]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.3
[0.0.2]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.2
[0.0.1]: https://github.com/qut-dmrc/buttermilk/releases/tag/v0.0.1
