# Development Standards

This document outlines the development standards and best practices for the Buttermilk project, derived from lessons learned during collaborative development sessions.

## Core Development Philosophy

### **Slow Down to Go Fast**
Spending time on analysis and planning prevents hours of debugging and thrashing between partial solutions. Understanding systems deeply prevents introducing new bugs and ensures changes are made for the right reasons.

### **Value Boring, Systematic Work**
- Comprehensive testing is more valuable than clever code
- Clear documentation prevents future confusion  
- Methodical analysis is more reliable than intuition
- Discipline in process is a skill that must be consciously practiced

## Mandatory Development Process

### Phase 1: Analysis (Before Any Code Changes)
**Required Outputs:**
- Written problem statement with clear scope
- System architecture map showing affected components
- Root cause identification (distinguish from symptoms)
- List of assumptions requiring validation

**Quality Gates:**
- [ ] Can explain the problem to someone else clearly
- [ ] Have identified the specific code/config causing the issue
- [ ] Understand how the fix will impact other parts of the system
- [ ] Have validated architectural assumptions with code inspection

### Phase 2: Planning
**Required Outputs:**
- GitHub issue for non-trivial changes (use provided template)
- Implementation plan with clear phases
- Success criteria defined upfront
- Risk assessment and rollback plan

**Quality Gates:**
- [ ] Plan addresses root cause, not just symptoms
- [ ] Changes are minimal and focused
- [ ] Success criteria are measurable
- [ ] Regression prevention strategy is clear

### Phase 3: Test-First Implementation
**Required Outputs:**
- Failing tests that demonstrate the current problem
- Implementation that makes tests pass
- Validation that existing tests still pass

**Quality Gates:**
- [ ] Tests fail with current implementation
- [ ] Tests pass after implementation
- [ ] No regressions in existing functionality
- [ ] Edge cases are covered

### Phase 4: Validation
**Required Outputs:**
- All success criteria met
- Documentation updated (if needed)
- Commit message explains "why" not just "what"

**Quality Gates:**
- [ ] Fix solves the original problem completely
- [ ] No new issues introduced
- [ ] Code is maintainable and well-documented

## Red Flags: When You're Moving Too Fast

Stop immediately if you find yourself:
- Proposing config changes without understanding data flow
- Making multiple small fixes instead of addressing root cause
- Suggesting "try this" without a systematic plan
- Modifying code without first writing tests that demonstrate the problem
- Saying "this should work" instead of "this will work because..."
- Jumping between different approaches without completing analysis

## Development Anti-Patterns to Avoid

### **Solution Shopping**
- **Problem**: Trying multiple quick fixes without understanding the root cause
- **Solution**: Complete Phase 1 analysis before proposing any solutions

### **Assumption Cascade**
- **Problem**: Building solutions on unvalidated assumptions about system behavior
- **Solution**: Validate each assumption with code inspection or testing

### **Scope Creep During Bug Fixes**
- **Problem**: "While we're here" refactoring during critical bug fixes
- **Solution**: One conceptual change per commit; improvements come later

### **Test-After Development**
- **Problem**: Writing tests to validate working code instead of proving the problem exists
- **Solution**: Write failing tests first, then implement until they pass

## Quality Standards

### **Code Changes**
- Single responsibility: Each change should solve one well-defined problem
- Minimal viable fix: Smallest change that addresses the root cause
- Clear reasoning: Commit messages explain why the change was necessary
- Backwards compatibility: Unless explicitly breaking changes are required

### **Testing Requirements**
- Problem demonstration: Tests that fail before the fix
- Edge case coverage: Handle missing fields, empty data, etc.
- Regression prevention: Existing functionality must continue working
- Integration validation: End-to-end workflows must be tested

### **Documentation Standards**
- Root cause analysis: Explain why the problem occurred
- Solution rationale: Why this approach vs. alternatives
- Impact assessment: What other systems are affected
- Future prevention: How to avoid similar issues

## Issue Template

When creating GitHub issues for development work, include:

```markdown
## Problem Statement
[Clear description of what's not working]

## Root Cause Analysis
[What specifically is causing the issue]

## Proposed Solution
[Minimal change that addresses root cause]

## Implementation Plan
- [ ] Phase 1: Analysis completed
- [ ] Phase 2: Tests written (failing)
- [ ] Phase 3: Implementation
- [ ] Phase 4: Validation

## Success Criteria
- [ ] [Specific, measurable outcomes]

## Risk Assessment
[What could go wrong and how to mitigate]

## Testing Strategy
[How to verify fix works and doesn't break anything]
```

## Cultural Standards

### **Embrace "I Don't Know Yet"**
It's always better to say "Let me analyze this systematically" than to guess. Unknown unknowns are the most dangerous - surface them early through systematic analysis.

### **Question First, Code Second**
Before implementing solutions, ask:
- Do I understand why this problem exists?
- What assumptions am I making about system behavior?
- What's the simplest change that solves the root cause?
- How will I know if this fix actually works?

### **Systematic Over Clever**
Prefer boring, well-tested solutions over clever, brittle ones. The goal is maintainable systems that future developers can understand and modify confidently.

## Enforcement

These standards are not suggestions - they are requirements for maintaining code quality and preventing the frustration of debugging poorly-understood systems.

**For Claude Code:** These guidelines override any tendency to rush to solutions. Always complete the analysis phase before proposing implementation approaches.

**For Human Developers:** Use these standards as a checklist during code reviews and development planning sessions.

**For Project Maintainers:** Enforce these standards through:
- Pull request templates that require phase completion
- Code review guidelines that check for systematic thinking
- Regular retrospectives on development process quality