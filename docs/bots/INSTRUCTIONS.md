<!-- This file should be read EVERY time. Keep it CONCISE and LIMITED to strictly required information. -->

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

# CRITICAL RULES for 🤖 LLM Agents: 

* ALWAYS follow the workflow. No exceptions.
* ASK FOR HELP WHEN YOU NEED IT. 
* DO NOT try to fix errors you do not fully understand.
* STOP if you get stuck. No workarounds.
* EVERYTHING is VALIDATED on rigorous academic research. DO NOT MAKE substantive decisions on your own. NO ASSUMPTIONS. No single-use scripts, no dummy examples; use ONLY our validated examples.
* Prioritise reproducibility & traceability.
* Flexible, modular design aimed at future Humanities, Arts, and Social Sciences researchers.

## WORKFLOW: Before Making Any Code Changes
1. **STOP**: Understand the full problem scope before proposing solutions. Read relevant documentation to understand the project goals and architecture. Check github issues for relevant past work and discussion; create a new issue if you cannot find an existing one.
2. **ANALYZE**: Map the system architecture and identify root causes
3. **PLAN**: Use github issues to track problems and document your plan with clear phases and validation criteria
4. **TEST**: Write failing tests that capture expected behavior
5. **IMPLEMENT**: Make minimal changes that solve the root cause
6. **DOCUMENT**: ALWAYS document your code with clear docstrings and comments
7. **VALIDATE**: Use the project's end-to-end debugging tools to ensure no regressions and all success criteria are met
8. **COMMIT** and **UPDATE GITHUB ISSUES**: Commit your changes in logical chunks and document each step in the appropriate github issue. If you're working independently, don't forget to file a pull request with your new changes!
9. **REFLECT**: Review your performance and update the `docs/bots` instructions if necessary to avoid mistakes.


## Key technical hints
- Repository: @qut-dmrc/buttermilk
- Owner: @nicsuzor
- Architecture: YAML configuration (/conf); FastAPI backend (buttermilk/api); Autogen-based LLM groupchat flows; web (/buttermilk/frontend/chat) and cli (/buttermilk/frontend/cli) frontends.
- Run python with `uv run ...`
- **Composable YAML Configuration**: Use Hydra (OmegaConf objects) exclusively for configuration
 

## REFLECTIVE and EXPERIMENTAL workflow

We are CONTINUOSLY refining our workflow. Agents ONLY remember the information we provide. Maintain the `docs/bots/` folder with essential information for robot developers.
- Use [INDEX.md](docs/bots/INDEX.md) to index and link to documentation and external tools.
- Update documents every time there is a key change (but not minor issues)
- Include all GENERAL important information developers need to understand, but REMOVE minor details or information that is specific to a particular task or scenario.
- Be CONCISE to save tokens.
- If you find conflicting information, ask the user for clarification, and then update the documents.



