# Issue: Enhanced Data Loader Column Mapping for OSB Configuration

## Summary

OSB data loading fails because the FileStorage column mapping logic incorrectly handles nested metadata mappings, resulting in empty metadata and missing field extraction. This affects any configuration using nested `metadata:` mappings in the `columns` field.

## Problem Description

### Current Behavior
When loading OSB data with the configuration in `conf/storage/osb.yaml`, the data loader produces:
- **Empty metadata**: `metadata={}` instead of extracted title, topics, standards, etc.
- **Raw content**: Content field contains JSON structure instead of extracted `fulltext`
- **Lost fields**: Unmapped fields from source data are completely discarded

### Expected Behavior
The data loader should:
- Extract `fulltext` field → Record `content`
- Extract structured metadata: `title`, `topics`, `standards`, `reasons`, `recommendations` → Record `metadata`
- Preserve unmapped fields as additional metadata

## Root Cause Analysis

### Architecture Overview
The current data loading system has a **dual approach**:
1. **New Unified Storage System** (recommended): `bm.get_storage(StorageConfig)` → `FileStorage`
2. **Legacy Data Loaders** (deprecated): `create_data_loader(DataSourceConfig)` → various loaders

**Key Finding**: Both `"gcs"` and `"file"` storage types use identical `FileStorage` implementation - there is no difference in data loading logic.

### Specific Bug Location
**File**: `buttermilk/storage/file.py:185-200`

**Current problematic code**:
```python
if self.config.columns:
    mapped_data = {}
    for new_key, old_key in self.config.columns.items():
        # Handle nested metadata mapping
        if new_key == 'metadata' and isinstance(old_key, dict):
            metadata = {}
            for meta_key, meta_source in old_key.items():
                if meta_source in data:
                    metadata[meta_key] = data[meta_source]
            mapped_data['metadata'] = metadata
        elif old_key in data:
            mapped_data[new_key] = data[old_key]
    
    # BUG: Complete replacement loses unmapped fields
    if mapped_data:
        data = mapped_data  # ← WRONG: Loses all unmapped fields
```

**Root Cause**: When `mapped_data` exists, the original `data` is completely replaced, losing all unmapped fields that should be preserved as metadata.

## Configuration Analysis

### Working Configurations
| Config | Type | Column Mapping | Status | Reason |
|--------|------|----------------|--------|--------|
| `tox.yaml` (drag) | file | Simple flat mapping | ✅ Works | Only maps a few fields |
| `tja_train.yaml` | file | No mapping | ✅ Works | No column mapping applied |

### Broken Configurations  
| Config | Type | Column Mapping | Status | Reason |
|--------|------|----------------|--------|--------|
| `osb.yaml` | gcs | Complex nested mapping | ❌ Broken | Complete data replacement bug |

### OSB Configuration Structure
```yaml
osb_json:
  type: gcs  # Uses FileStorage (same as "file")
  path: gs://prosocial-public/osb/03_osb_fulltext_summaries.json
  columns:
    record_id: record_id
    content: fulltext  # Extract fulltext field
    metadata: 
      title: title
      description: content  # Different field than fulltext!
      result: result
      type: type
      location: location
      case_date: case_date
      topics: topics
      standards: standards  
      reasons: reasons
      recommendations: recommendations
      job_id: job_id
      timestamp: timestamp
```

### Expected OSB Data Flow
**Source JSON**:
```json
{
  "record_id": "BUN-QBBLZ8WI",
  "fulltext": "This is the main content for vector processing...",
  "title": "Mention of Al-Shabaab",
  "content": "The first post included a picture...",  
  "result": "leave up",
  "type": "summary",
  "topics": ["War and conflict", "Dangerous individuals and organizations"],
  "standards": ["Dangerous Individuals and Organizations policy"],
  "reasons": ["The policy prohibits content...", "In the first post..."],
  "recommendations": ["Enhance training...", "Add criteria..."]
}
```

**Expected Record**:
```python
Record(
  record_id="BUN-QBBLZ8WI",
  content="This is the main content for vector processing...",  # from fulltext
  metadata={
    "title": "Mention of Al-Shabaab",
    "description": "The first post included a picture...",  # from content field
    "result": "leave up",
    "type": "summary", 
    "topics": ["War and conflict", "Dangerous individuals and organizations"],
    "standards": ["Dangerous Individuals and Organizations policy"],
    "reasons": ["The policy prohibits content...", "In the first post..."],
    "recommendations": ["Enhance training...", "Add criteria..."]
  }
)
```

## Impact Assessment

### Currently Broken
- ❌ OSB data loading (affects vector search and RAG functionality)
- ❌ Any configuration with nested `metadata:` mappings
- ❌ Configurations that don't map all source fields explicitly

### Currently Working
- ✅ Simple flat column mappings (tox/drag config)
- ✅ Configurations with no column mapping (tja config)
- ✅ BigQuery storage (different implementation)
- ✅ ChromaDB storage (different implementation)

## Solution Plan

### Phase 1: Fix Core Bug (High Priority)
1. **Fix FileStorage column mapping logic**:
   ```python
   # Change from:
   if mapped_data:
       data = mapped_data  # Complete replacement
   
   # To:
   if mapped_data:
       data = {**data, **mapped_data}  # Merge with original
   ```

2. **Create comprehensive unit tests** covering:
   - Simple flat mappings (existing functionality)
   - Nested metadata mappings (OSB use case)
   - Mixed mappings (some fields mapped, others preserved)
   - Edge cases (missing fields, empty mappings)

### Phase 2: Validation (Medium Priority)
3. **Test all existing configurations**:
   - Verify `tox.yaml` still works (regression test)
   - Verify `tja_train.yaml` still works (regression test)
   - Verify `osb.yaml` now works (fix validation)

4. **Integration testing**:
   - Test OSB vector database creation pipeline
   - Test enhanced RAG agents with properly loaded OSB data
   - Verify no performance regressions

### Phase 3: Documentation & Cleanup (Low Priority)
5. **Update documentation**:
   - Document nested metadata mapping capabilities
   - Provide examples of complex column configurations
   - Clarify unified storage system as primary approach

6. **Legacy code cleanup**:
   - Mark deprecated loaders clearly
   - Consider removing unused legacy data loaders (separate issue)

## Test Cases Required

### Unit Tests (`tests/storage/test_file_storage_column_mapping.py`)

1. **Simple Mapping Test**:
   ```python
   config = {"content": "text", "ground_truth": "expected"}
   # Should map specified fields, preserve others as metadata
   ```

2. **Nested Metadata Mapping Test**:
   ```python
   config = {
       "content": "fulltext", 
       "metadata": {"title": "title", "desc": "content"}
   }
   # Should extract nested metadata structure correctly
   ```

3. **Mixed Mapping Test**:
   ```python
   config = {
       "content": "text",
       "metadata": {"title": "title"}, 
       "ground_truth": "expected"
   }
   # Should handle both direct field mapping and nested metadata
   ```

4. **OSB Real Data Test**:
   ```python
   # Use actual OSB JSON structure from the data file
   # Validate all expected fields are extracted correctly
   ```

5. **Regression Tests**:
   ```python
   # Ensure tox and tja configurations still work
   # Verify no existing functionality is broken
   ```

### Integration Tests (`tests/flows/test_osb_data_loading.py`)

1. **End-to-End OSB Loading**:
   - Load OSB data using actual `osb.yaml` config
   - Verify Record fields are populated correctly
   - Test vector database creation pipeline

2. **Configuration Compatibility**:
   - Test all storage configs: `osb.yaml`, `tox.yaml`, `tja_train.yaml`
   - Verify backward compatibility

## Success Criteria

### Functional Requirements
- [ ] OSB data loads with populated metadata containing all structured fields
- [ ] Content field contains clean fulltext (not JSON structure)
- [ ] Vector database creation works correctly with OSB data
- [ ] Enhanced RAG agents can search across OSB metadata fields
- [ ] Existing tox and tja configurations continue to work

### Non-Functional Requirements  
- [ ] No performance degradation in data loading
- [ ] Comprehensive test coverage (>90% for modified code)
- [ ] Clear documentation of nested metadata mapping capabilities
- [ ] Backward compatibility maintained

## Risk Assessment

### Low Risk
- **Simple fix**: The bug is localized to one line of code
- **Clear root cause**: Problem is well-understood and isolated
- **Existing tests**: Current functionality can be regression tested

### Mitigation Strategies
- **Comprehensive testing**: Test all existing configurations before and after fix
- **Incremental deployment**: Fix and test one config at a time
- **Rollback plan**: Simple to revert if issues arise

## Dependencies

### Internal
- Enhanced RAG agents (benefit from fix)
- Vector database pipeline (requires fix)
- Record validation system (may need updates)

### External
- No external dependencies
- Only affects internal data loading logic

---

**Priority**: High
**Complexity**: Low
**Effort**: 2-3 hours (fix + comprehensive testing)
**Risk**: Low (localized change with clear solution)