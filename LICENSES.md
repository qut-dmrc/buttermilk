# Dependency license audit

This document records the license audit performed against buttermilk's resolved dependency tree to verify compatibility with buttermilk's GPL-3.0-or-later license.

## Methodology

```bash
uv sync --extra dev --extra ml
uv run --with pip-licenses pip-licenses --format=markdown --with-urls --order=license > LICENSES-detailed.md
```

The audit covers all packages resolved from `pyproject.toml` (`dependencies` + `[project.optional-dependencies]`) plus their transitive dependencies.

Regenerate this report after any dependency change: `uv lock && rerun the command above`.

## Verdict

**All resolved dependencies are compatible with GPL-3.0-or-later.** Distribution of buttermilk as a combined work with these dependencies does not create license conflict.

## License distribution

| License family | Packages | Compatible with GPL-3.0-or-later? |
|---|---|---|
| MIT / MIT License | 124 | Yes — permissive, GPL-3-compatible |
| Apache-2.0 / Apache Software License | 124 | Yes — GPL-3 explicitly compatible (GPLv2 incompatibility resolved in v3) |
| BSD-2-Clause / BSD-3-Clause / BSD | 59 | Yes — permissive |
| Mozilla Public License 2.0 | 3 | Yes — MPL-2.0 is explicitly GPL-compatible |
| PSF (Python Software Foundation) | 4 | Yes — explicitly GPL-compatible |
| ISC License | 2 | Yes — permissive |
| GPL-3.0-or-later | 2 | Yes — same license |
| GPL-2.0-or-later | 1 (`Unidecode`) | Yes — "or later" allows GPL-3 |
| Unlicense | 1 | Yes — public domain dedication |
| Dual / multi-license expressions | several | Yes — at least one compatible option in each |
| Unknown / metadata missing | a few | Manually verified — see below |

## Anomalies investigated

### `poppler-utils` (REMOVED)

The PyPI package `poppler-utils==0.1.0` was declared as a direct dependency and ships GPL-2-only metadata, which would have been incompatible with GPL-3-or-later.

Investigation showed the package was **never imported** anywhere in `buttermilk/`. Buttermilk uses `pdftotext` from the *system* `poppler-utils` binary via `subprocess` (see `buttermilk/processors/bash.py`), not the Python package. The PyPI dependency was a misunderstanding and has been removed. A comment in `pyproject.toml` directs users to install the system binary via their package manager.

### Packages with `License: UNKNOWN` in metadata

These have known licenses confirmed via their source repositories; the UNKNOWN tag reflects missing PyPI classifier metadata, not license ambiguity:

| Package | Actual license | Source |
|---|---|---|
| `google-crc32c` | Apache-2.0 | Google open-source standard, confirmed at https://github.com/googleapis/python-crc32c |
| `sentencepiece` | Apache-2.0 | Confirmed at https://github.com/google/sentencepiece/blob/master/LICENSE |
| `antlr4-python3-runtime` | BSD-3-Clause | ANTLR has been BSD-3-Clause since v4 |

### Multi-license expressions

A handful of packages publish SPDX expressions like `MPL-2.0 AND (Apache-2.0 OR MIT)`. Each contains at least one GPL-3-compatible component, so the combination is compatible.

## What this audit does NOT cover

- **Runtime-installed extras**: if a user installs additional optional dependencies (e.g. `[ml]`, model-specific provider SDKs), those should be re-audited.
- **System binaries invoked via subprocess** (e.g. `pdftotext`, `git`, `gcloud`): these are not linked into the buttermilk distribution, so their licenses do not affect distribution compatibility.
- **Hosted services accessed over the network** (LLM APIs, BigQuery, etc.): not a combined-work concern; usage governed by each provider's terms.

## What to do if a future dependency change introduces an incompatible license

1. **Stop.** Do not publish the next release until resolved.
2. Identify whether the package is genuinely needed. Often (as with `poppler-utils` above) it is not.
3. If genuinely needed, look for an equivalent under a compatible license.
4. If no alternative exists, escalate: either move buttermilk to a compatible license, or move the dep into an optional extra that users opt into knowingly.

GPL-3.0-or-later incompatible licenses to watch for include: GPL-2-only, CDDL, EPL-1.0, MPL-1.1, BSD-4-Clause (advertising), and proprietary or source-available licenses with field-of-use restrictions.
