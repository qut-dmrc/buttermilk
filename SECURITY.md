# Security Policy

## Supported Versions

Only the most recent release on PyPI receives security fixes. Buttermilk follows semantic versioning via `setuptools_scm`; see `CHANGELOG.md` for the latest version.

## Reporting a Vulnerability

**Please do not file public issues for security vulnerabilities.**

Report security issues privately by either:

1. Emailing **n.suzor@qut.edu.au** with subject line `[buttermilk security]`, or
2. Using GitHub's [private vulnerability reporting](https://github.com/qut-dmrc/buttermilk/security/advisories/new) on this repository.

### What to include

- A description of the vulnerability and its potential impact.
- Steps to reproduce, ideally with a minimal example.
- The version of buttermilk and Python you observed it on.
- Whether you have a proposed fix.

### Response expectations

- Initial acknowledgement: within 5 business days.
- Triage and severity assessment: within 10 business days.
- Fix timeline: depends on severity. We will coordinate disclosure with you.

## Scope

In scope:

- The published `buttermilk` package on PyPI.
- The Docker images published from `containers/`.
- Authentication, credential handling, and data-storage paths.
- Dependencies used in the default install (we may forward upstream).

Out of scope:

- Issues in third-party dependencies whose upstream has a public advisory — report to them directly.
- Misconfigurations in user-controlled YAML / Hydra configs, secrets management, or cloud-provider IAM.
- Findings that require an attacker to already have full local or admin access to the user's machine.

## Research-data integrity

Buttermilk is research software used by HASS scholars. Issues that compromise the *integrity* or *traceability* of research outputs (e.g. silent logging failures, dropped traces, broken reproducibility) are treated as security-class issues even when not exploitable in the traditional sense. Please report these the same way.
