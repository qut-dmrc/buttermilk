# Tests Requiring Management Decisions

This document tracks tests where expected behavior is unclear and requires management/domain expert decisions.

## tests/data/test_records_openaimessages.py

### test_as_openai_message_with_text
- **Issue**: API changed from returning dict to returning UserMessage object
- **Old behavior**: `as_message()` returned dict that could be subscripted
- **New behavior**: Returns UserMessage object (not subscriptable)
- **Question**: Should test be updated to use object attributes, or should API return dict?
- **File**: tests/data/test_records_openaimessages.py:156

### test_as_openai_message_with_media
- **Issue**: Image format validation changed
- **Old behavior**: Accepted PIL Image objects directly
- **New behavior**: Expects dict or Image instance, rejects PIL objects
- **Question**: Should we wrap PIL images before passing, or should validation accept PIL?
- **File**: tests/data/test_records_openaimessages.py:137

### test_as_openai_message_no_media_no_text
- **Issue**: No longer raises OSError for empty content
- **Old behavior**: Raised OSError when no media or text provided
- **New behavior**: Does not raise error
- **Question**: Should empty content be an error, or is silent handling OK?
- **File**: tests/data/test_records_openaimessages.py:162

## tests/00initial/test_bm_singleton.py

### test_singleton_instance, test_singleton_between_modules, etc.
- **Issue**: Singleton behavior expectations unclear with new BMAccessor proxy
- **Old behavior**: Expected `is` identity checks to work
- **New behavior**: BMAccessor proxy breaks identity checks
- **Question**: How should singleton behavior work with the accessor pattern?
- **Files**: Multiple tests in tests/00initial/test_bm_singleton.py

## tests/00initial/test_config.py

### test_has_test_info, test_save_dir
- **Issue**: Assertion failures about expected config values
- **Question**: Need to review what actual config values should be
- **File**: tests/00initial/test_config.py

## tests/00initial/test_init.py

### test_short_form_cli, test_short_form_nb
- **Issue**: RuntimeError: Project name required
- **Old behavior**: Project name was optional or had default
- **New behavior**: Project name is required field
- **Question**: Should these convenience functions provide a default project name?
- **Files**: tests/00initial/test_init.py:test_short_form_cli, test_short_form_nb

---

## Resolution Process

For each item above:
1. Domain expert reviews old vs new behavior
2. Decides which is correct
3. Updates test or reverts code change
4. Removes item from this list
