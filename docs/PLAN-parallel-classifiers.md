# Architectural Plan: Parallel Classifier Integration

**Problem**: Add Cope toxicity model to reliability pipeline alongside 7 LLM models
**Date**: 2024-11-28
**Status**: DRAFT - Awaiting Review

## Problem Statement

The reliability study pipeline currently runs:
- 7 LLM models × 8 prompt templates × 5 repetitions = 280 classifications per record
- Using `VariantProcessor` with `LLMCore` as the processor

Need to add:
- Cope toxicity model × 5 repetitions = 5 additional classifications per record
- Running **in parallel** with LLM models (not sequentially chained)

## Current Architecture

```
Source → [VariantProcessor(LLMCore, 7 models × 8 templates × 5 runs)] → Output
```

**Key Constraint**: Pipeline processors are **chained sequentially** - output from stage N becomes input to stage N+1. This means adding Cope as a second processor would cause it to process all 280 LLM outputs, not the original record.

## Design Options Considered

### Option A: ParallelProcessor (RECOMMENDED)
New processor that wraps multiple sub-processors, running all on the **same input** in parallel.

**Pros**:
- Clean separation of concerns
- Composable - any processors can be parallelized
- Uses existing VariantProcessor infrastructure
- No changes to existing working code
- Follows Processor protocol

**Cons**:
- One new abstraction layer

### Option B: Extend VariantProcessor for multiple processor classes
Allow VariantProcessor to take multiple `processor_obj` values.

**Pros**:
- Single concept for "run variants in parallel"

**Cons**:
- Complicates working code
- Different processor classes have different interfaces
- Awkward parameter handling

### Option C: Unified Classifier abstraction
Create common interface for all classifiers (LLMCore, ToxicityModel).

**Pros**:
- Cleanest long-term if we have many classifier types

**Cons**:
- Bigger refactor
- Premature optimization - only have 2 classifier types

### Option D: Separate pipelines
Run LLMs and Cope in separate pipeline runs, merge after.

**Pros**:
- Zero code changes

**Cons**:
- Coordination overhead
- Duplicate source processing
- Results scattered across runs

## Recommended Solution: ParallelProcessor

### Architecture

```
Source → [ParallelProcessor] → Output
              ├── VariantProcessor(LLMCore, 7×8×5)
              └── VariantProcessor(Cope, 1×1×5)
```

### ParallelProcessor Design

```python
class ParallelProcessor(BaseModel):
    """Run multiple processors in parallel on the SAME input record.

    Unlike chained processors where outputs flow sequentially,
    ParallelProcessor runs all sub-processors on the original input.
    """
    processors: list[Any]  # Sub-processors to run in parallel
    fail_on_error: bool = False

    async def process(self, record, *, processor_stage, **kwargs):
        # Create async tasks for all processors
        tasks = [proc.process(record, ...) for proc in self.processors]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            for output in await coro:
                yield output  # with parallel metadata added
```

### Configuration

```yaml
processors:
  - _target_: buttermilk.processors.ParallelProcessor
    processors:
      # LLM-based classifiers (7 models × 8 prompts × 5 runs = 280)
      - _target_: buttermilk.processors.VariantProcessor
        processor_obj: buttermilk._core.llm_core.LLMCore
        variants:
          model:
            - llama4maverick
            - gpt5mini
            - claude45haiku
            - gemini-pro
            - gemini-flash
            - gpt-4o
            - gpt-oss-safeguard-20b
          template:
            - list_no-cot_zero-shot
            - list_no-cot_few-shot
            - list_cot_zero-shot
            - list_cot_few-shot
            - prose_no-cot_zero-shot
            - prose_no-cot_few-shot
            - prose_cot_zero-shot
            - prose_cot_few-shot
        num_runs: 5
        parameters:
          output_model: src.contracts.HateSpeechClassification
          output_col: prediction
          fail_on_unfilled_parameters: true

      # API-based classifier (Cope × 5 runs = 5)
      - _target_: buttermilk.processors.VariantProcessor
        processor_obj: buttermilk.toxicity.Cope
        variants: {}  # No parameter variants
        num_runs: 5   # Just repetitions for reliability
```

### Key Insight: VariantProcessor handles num_runs for Cope

Looking at `ProcessorVariants.get_configs()`:
```python
variant_combinations = expand_dict(self.variants) if self.variants else [{}]

for _ in range(self.num_runs):
    for variant_params in variant_combinations:
        # ... generates configs
```

With `variants: {}` and `num_runs: 5`, this generates 5 identical Cope instances - exactly what we need for reliability repetitions.

## Implementation Steps

1. **Verify Cope model works** (already done in buttermilk)
   - Added `Cope` class to `toxicity.py`
   - Added to `__init__.py` exports
   - Has `process()` method via ToxicityModel base class

2. **Create ParallelProcessor** (already started)
   - File: `buttermilk/processors/parallel.py`
   - Exports in `buttermilk/processors/__init__.py`

3. **Write tests for ParallelProcessor**
   - Unit test: verify parallel execution
   - Unit test: verify same input to all processors
   - Unit test: error handling (fail_on_error modes)

4. **Update reliability pipeline config**
   - Update `/home/nic/src/reliability/config/run/pipeline.yaml`

5. **Test with limit=2**
   - Run: `uv run bm --config-dir=.../config --config-name=reliability +run=pipeline`
   - Verify outputs include both LLM and Cope results

## Output Data Shape

Each input record produces:
- 280 outputs from LLM models (with variant metadata)
- 5 outputs from Cope (with parallel + variant metadata)

Metadata structure:
```python
record.metadata = {
    "parallel": {
        "processor_index": 0,  # Which ParallelProcessor branch
        "total_processors": 2,
        "processor_class": "VariantProcessor",
    },
    "variant": {
        "index": 42,  # Which variant within VariantProcessor
        "total": 280,
        "processor_class": "LLMCore",
    },
    # ... processor-specific outputs (prediction, scores, etc.)
}
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Cope credentials not configured | Fail-fast with clear error message |
| Cope API rate limits | ParallelProcessor runs async, natural backpressure |
| Cope response format mismatch | ToxicityModel.interpret() validates response |
| Parallel execution overhead | asyncio.as_completed() streams results efficiently |

## Success Criteria

1. Pipeline runs with `limit=2` without errors
2. Output contains both LLM and Cope results
3. Each record has 285 total outputs (280 LLM + 5 Cope)
4. Metadata correctly identifies source processor
5. Cope credentials work from buttermilk secrets

## Questions for Review

1. Should ParallelProcessor live in buttermilk or reliability project?
   - **Recommendation**: buttermilk - it's a general-purpose pattern

2. Should we support nested ParallelProcessors?
   - **Recommendation**: Not initially, add if needed

3. How should we handle partial failures?
   - **Recommendation**: `fail_on_error=False` default, log and continue
