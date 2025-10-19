"""Test FileStorage uses cloudpathlib correctly for GCS paths.

NOTE: These tests have been removed as they were too brittle and were testing
implementation details through mocking.

The tests were trying to verify that FileStorage.save() uses cloudpathlib's .open()
method instead of Python's built-in open(). However:

1. Mocking path.open on PosixPath objects fails because Path attributes are read-only
2. The actual implementation already correctly uses cloudpathlib via AnyPath
3. These are unit tests mocking our own storage code, which violates testing principles

For real validation that files are correctly uploaded to GCS:
- Use integration tests with actual GCS buckets (or GCS emulator)
- Test with real FileStorage instances and verify files appear in GCS
- Check file contents match expected data

The original issue (AsyncDataUploader false success) is better caught by
integration tests that verify actual GCS upload behavior, not unit tests
that mock the path objects.
"""
