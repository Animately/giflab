# Planned Features Documentation System

This directory contains scope documents for planned features in GifLab. These documents follow a standardized format that enables Claude Code to easily consume, understand, and update implementation progress.

## Table of Contents

- [Document Structure](#document-structure)
  - [Required Frontmatter](#required-frontmatter)
  - [Phase-Based Organization](#phase-based-organization)
  - [Status Legend](#status-legend)
- [Document Types](#document-types)
- [Pipeline Creation vs Filtering Approach](#pipeline-creation-vs-filtering-approach)
- [Claude Code Integration](#claude-code-integration)
- [Best Practices](#best-practices)
- [Template Usage](#template-usage)
- [Updating Progress](#updating-progress)
- [Examples](#examples)

## Document Structure

### Required Frontmatter
Every scope document must begin with YAML frontmatter containing these required keys with their allowed values:

```yaml
---
name: Feature Name
priority: low|medium|high  # Priority level for implementation
size: tiny|small|medium|large  # Estimated complexity/effort
status: planning|ready|in-progress|testing|complete  # Current state
owner: @username  # GitHub username of responsible person
issue: GitHub issue number or "N/A"  # Associated issue tracking
---
```

#### Example Frontmatter
```yaml
---
name: Modular Experiment Presets
priority: high
size: medium
status: in-progress
owner: @lachlants
issue: "#245"
---
```

### Phase-Based Organization

All scope documents must be organized into clearly defined phases with trackable completion status:

#### Phase Tracking Format
```markdown
### Phase N: Phase Name ⏳ PLANNED
**Progress:** X% Complete
**Current Focus:** Brief description of current work

#### Subtask N.1: Subtask Name ✅ COMPLETE
- [x] Completed task
- [ ] Pending task
- [x] Another completed task

#### Subtask N.2: Another Subtask Name ⏳ PLANNED
- [ ] Pending task
- [ ] Another pending task
```

#### Subtask Numbering Convention
- Use hierarchical numbering: `Phase.Subtask` (e.g., 1.1, 1.2, 2.1, 2.2, etc.)
- Phase numbers correspond to the main phase number
- Subtask numbers increment sequentially within each phase
- This enables precise referencing and progress tracking

### Status Legend

| Icon | Status | Description |
|------|--------|-------------|
| ⏳ | PLANNED | Not yet started |
| 🔄 | IN PROGRESS | Currently being worked on |
| ✅ | COMPLETE | Finished successfully |
| ⚠️ | BLOCKED | Cannot proceed due to dependency |
| ❌ | CANCELLED | No longer needed/feasible |

## Document Types

### Small Updates (tiny/small size)
- **3-5 phases maximum**
- **Simple linear progression**
- **Minimal dependencies**

Example phases:
1. Planning & Design
2. Implementation
3. Testing & Documentation

### Large Updates (medium/large size)
- **5-8 phases maximum**
- **Complex interdependencies**
- **Multiple implementation stages**

Example phases:
1. Planning & Requirements Analysis
2. Architecture Design
3. Core Implementation
4. Integration & Interface Development
5. Testing & Validation
6. Documentation & Polish
7. Deployment & Handoff

## Pipeline Creation vs Filtering Approach

### Modern Approach: Targeted Pipeline Creation
For experiment preset systems, focus on **creating specific pipelines** rather than filtering from a full matrix:

#### Variable Scopes & Locked Implementations
```markdown
### Preset Definition Structure
- **Variable Slots**: Specify which algorithm types vary (frame, color, lossy)
- **Variable Scope**: List exact algorithms/options to test for variable slots
- **Locked Implementations**: Specify exact algorithms/settings for non-variable slots

Example:
- Frame Slot: [Variable] → All available frame reduction algorithms
- Color Slot: [Locked] → FFmpeg color reduction at 32 colors
- Lossy Slot: [Locked] → animately-advanced compression at level 40
```

## Claude Code Integration

Claude Code (AI assistant) automatically updates scope documents during implementation. Human contributors should also follow these patterns when manually updating progress.

### Status Updates
Update phase status immediately upon completion:

```markdown
### Phase 2: Implementation ✅ COMPLETE
**Progress:** 100% Complete
**Completed:** 2025-01-15
```

### Progress Tracking
Use specific completion percentages and current focus descriptions:

```markdown
### Phase 3: Testing 🔄 IN PROGRESS
**Progress:** 60% Complete
**Current Focus:** Integration test development
**Remaining:** Performance validation, edge case testing
```

### Completion Criteria
Each phase should have clear, measurable completion criteria:

```markdown
#### Completion Criteria
- [ ] All unit tests passing (>95% coverage)
- [ ] Integration tests implemented and passing
- [ ] Performance benchmarks meet targets
- [ ] Documentation updated
```

## Best Practices

### Phase Granularity
- Phases should be 1-3 days of work maximum
- Each phase should have 3-8 subtasks
- Subtasks should be 1-4 hours of work each

### Dependency Management
- Clearly mark inter-phase dependencies
- Use BLOCKED status when dependencies prevent progress
- Include dependency resolution in planning phases

### Documentation Standards
- Update status immediately upon phase completion
- Include specific completion dates and metrics
- Document key decisions and architectural choices
- Maintain clear current focus descriptions

## Template Usage

When creating new scope documents:

1. Copy frontmatter template and fill in project-specific details
2. Choose appropriate phase structure based on project size
3. Define clear completion criteria for each phase
4. Set up progress tracking with percentages and status indicators
5. Plan for Claude Code updates with specific checkpoint tasks

## Updating Progress

### When You Complete a Subtask:
1. Update the checkbox: `- [x] Completed task`
2. Adjust the phase's **Progress:** percentage
3. Update **Current Focus:** to reflect next priority (e.g., "Moving to Subtask 1.3")
4. If the phase hits 100%, change its status icon to ✅ COMPLETE
5. Add completion date: `**Completed:** 2025-01-15`

### When You Start a New Phase:
1. Change status from ⏳ PLANNED to 🔄 IN PROGRESS
2. Set initial **Progress:** percentage (typically 0-10%)
3. Set **Current Focus:** to first major subtask (e.g., "Working on Subtask 2.1")
4. Mark first subtask as in progress if applicable

### Referencing Subtasks:
- Use the numbered format when discussing specific tasks: "Subtask 3.2 is blocked"  
- This enables precise communication about progress and dependencies
- Claude Code can efficiently update specific subtasks using these references

## Examples

See the modular-experiment-presets.md file in this directory for a comprehensive example of a large-scale feature scope document following these standards (when available).

---

*This documentation system enables efficient collaboration between human developers and Claude Code by providing clear structure, progress tracking, and implementation guidance.*