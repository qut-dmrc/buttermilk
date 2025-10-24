# Project-Specific Bot Configuration

This directory contains **OPTIONAL** project-specific bot configuration that supplements the core academicOps framework.

## Relationship to academicOps

**Core Framework**: The [academicOps repository](https://github.com/nicsuzor/academicOps) provides the foundational bot configuration, skills, agents, and workflows used across all projects.

**This Directory**: Contains project-specific customizations and extensions for the Buttermilk project only. These files are:

- **Optional**: Projects work without this directory by using only academicOps defaults
- **Additive**: Supplement (don't replace) academicOps configuration
- **Project-Specific**: Only applicable to this project, not portable to others

## Directory Structure

```
bot/
├── README.md           # This file - explains optional nature
├── docs/              # Project-specific documentation chunks
│   └── _CHUNKS/       # Modular documentation referenced by CLAUDE.md
└── [other files]      # Project-specific bot customizations
```

## When to Add Files Here

Add files to `bot/` when you need:

- Project-specific documentation chunks
- Custom skills unique to this project
- Project-specific agent configurations
- Overrides of academicOps defaults (use sparingly)

## When NOT to Add Files Here

Do NOT add to `bot/` if the content:

- Applies to all academicOps projects → Contribute to academicOps repo
- Is temporary/experimental → Use `.gitignore` or local files
- Documents the project itself → Use `docs/` (not `bot/docs/`)

## Installation

The `bot/` directory is automatically used by Claude Code when present. No installation required.

If this directory is missing, the project will use only the core academicOps configuration from the parent framework.

## See Also

- [academicOps Framework](https://github.com/nicsuzor/academicOps) - Core bot configuration
- `/docs/` - Project documentation (for humans)
- `CLAUDE.md` - Project-specific agent instructions
