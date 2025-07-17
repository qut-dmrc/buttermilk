# Documentation Maintenance Plan

This document outlines how to keep Buttermilk documentation accurate and up-to-date.

## Documentation Structure

### Target Audiences
1. **HASS Researchers** - Primary users, need practical guidance
2. **LLM Developers** - Bots working on the codebase
3. **Software Developers** - Human contributors and maintainers

### Documentation Hierarchy
```
docs/
├── bots/                   # LLM developer knowledge bank
│   ├── README.md          # Essential patterns and guidelines  
│   ├── config.md          # Complete configuration reference
│   ├── debugging.md       # Tools and troubleshooting
│   ├── development.md     # Systematic development workflow
│   ├── goals.md           # Project philosophy
│   ├── map.md             # Project structure overview
│   └── techstack.md       # Architecture and technology
├── getting-started/        # First-time user onboarding
├── user-guide/            # Practical usage for researchers
├── developer-guide/       # Human developer resources
└── reference/             # API and concept references
```

## Maintenance Responsibilities

### When Code Changes
- **Agent Interface Changes**: Update `docs/developer-guide/creating-agents.md`
- **Configuration Changes**: Update both `docs/user-guide/configuration.md` and `docs/bots/config.md`
- **New Tools/Features**: Update relevant sections across all guides
- **API Changes**: Update `docs/user-guide/api-reference.md`

### Documentation Ownership
- **docs/bots/**: LLM developers (automatic via this cleanup)
- **docs/user-guide/**: Focus on HASS researcher needs
- **docs/developer-guide/**: Human developer experience
- **docs/getting-started/**: Clear onboarding path

## Quality Standards

### Examples Must Work
- All code examples should be tested
- All CLI commands should be verified
- All configuration snippets should be valid YAML
- All file paths should exist

### Consistency Requirements
- Use current agent interface (`message: AgentInput`)
- Reference actual configuration files from `conf/`
- Use working flow names (`trans`, `osb`, `zot`, `tox`)
- Cross-reference between sections appropriately

### Content Guidelines
- **Be Concise**: Respect reader's time and tokens
- **Be Practical**: Focus on real-world usage
- **Be Current**: Update when code changes
- **Be Linked**: Create clear navigation paths

## Update Process

### For Major Changes
1. **Identify Impact**: Which docs need updates?
2. **Update Primary**: Fix the main documentation
3. **Update Cross-References**: Fix related sections
4. **Test Examples**: Verify all examples work
5. **Review Links**: Check internal references

### For Minor Changes
1. **Update Immediately**: Don't let small issues accumulate
2. **Check Context**: Ensure changes make sense in surrounding content
3. **Verify Commands**: Test any changed CLI examples

## Common Maintenance Tasks

### Quarterly Review
- [ ] Test all getting-started examples
- [ ] Verify all internal links work
- [ ] Check configuration examples against actual files
- [ ] Review for outdated references

### After Major Releases
- [ ] Update version-specific information
- [ ] Review architectural descriptions
- [ ] Update API documentation
- [ ] Check external links

### When Adding Features
- [ ] Add to appropriate user guide section
- [ ] Update bot knowledge bank if relevant
- [ ] Add to getting-started if it's a common task
- [ ] Update cross-references

## Red Flags - Documentation Drift

### Signs of Outdated Docs
- Examples that don't work
- References to removed features
- Inconsistent terminology
- Broken internal links
- Multiple conflicting explanations

### Prevention Strategies
- **Link Code Reviews to Doc Updates**
- **Automated Link Checking** (future enhancement)
- **Example Testing** in CI/CD
- **Regular Audits** by documentation team

## Documentation Anti-Patterns

### Avoid These Mistakes
- **Duplicating Information**: Keep single source of truth
- **Generic Examples**: Use real, working configurations
- **Stale Cross-References**: Update links when moving content
- **Overwhelming Detail**: Match depth to audience needs
- **Missing Context**: Explain when and why to use features

### Instead, Do This
- **Cross-Reference Related Content**: Link to authoritative sources
- **Use Real Examples**: Pull from actual working configurations
- **Maintain Clear Hierarchy**: Separate concerns by audience
- **Keep Context Clear**: Explain the "why" not just the "how"

## Emergency Procedures

### When Documentation is Critically Wrong
1. **Immediate Fix**: Correct the immediate problem
2. **Scope Assessment**: Check for similar issues elsewhere
3. **Root Cause Analysis**: Why did this happen?
4. **Process Improvement**: Prevent recurrence

### When Major Restructuring is Needed
1. **Plan Carefully**: Don't break existing links unnecessarily
2. **Migrate Content**: Don't lose valuable information
3. **Update All References**: Check both internal and external links
4. **Communicate Changes**: Update any documentation that references the structure

## Success Metrics

### Documentation Quality
- New users can complete getting-started without help
- LLM developers can find answers in bot knowledge bank
- Examples work when copy-pasted
- Cross-references lead to helpful information

### Maintenance Success
- Documentation stays current with code changes
- Inconsistencies are caught and fixed quickly
- No duplicate information across different files
- Clear ownership and update responsibilities

## Tools and Automation

### Current Tools
- Manual review during development
- Git hooks for documentation changes
- Systematic audits via LLM assistance

### Future Enhancements
- Automated link checking
- Example testing in CI/CD
- Documentation coverage metrics
- Automated cross-reference validation

## Review Schedule

### Weekly
- Review recent code changes for doc impact
- Check for new issues mentioning documentation

### Monthly  
- Review user feedback on documentation
- Check analytics for most-visited pages

### Quarterly
- Comprehensive audit of all documentation
- Review and update this maintenance plan
- Check for new documentation needs

Remember: Good documentation saves more time than it costs. Invest in keeping it accurate and useful.