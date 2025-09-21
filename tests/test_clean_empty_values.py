#!/usr/bin/env python3
"""Test the clean_empty_values function to ensure it doesn't modify original data."""

import copy

from buttermilk.utils.utils import clean_empty_values


def test_clean_empty_values():
    """Test that clean_empty_values doesn't modify the original data structure."""
    
    # Test data with various empty values
    original_data = {
        "a": 1,
        "b": None,
        "c": [],
        "d": {"e": "", "f": 2},
        "g": [1, None, [], {"h": None}],
        "i": {"j": {"k": {"l": None}}},
        "m": [{"n": None}, {"o": 3}]
    }
    
    # Make a deep copy to verify original isn't modified
    data_before = copy.deepcopy(original_data)
    
    # Clean the data
    cleaned = clean_empty_values(original_data)
    
    # Verify original data is unchanged
    assert original_data == data_before, "Original data was modified!"
    
    # Verify cleaned data is correct
    expected = {
        "a": 1,
        "d": {"f": 2},
        "g": [1],
        "m": [{"o": 3}]
    }
    
    assert cleaned == expected, f"Expected {expected}, got {cleaned}"
    
    # Test with list as root
    original_list = [1, None, [], {"a": None}, {"b": 2}]
    list_before = copy.deepcopy(original_list)
    cleaned_list = clean_empty_values(original_list)
    
    assert original_list == list_before, "Original list was modified!"
    assert cleaned_list == [1, {"b": 2}], f"Expected [1, {{'b': 2}}], got {cleaned_list}"
    
    print("✓ All tests passed! Function doesn't modify original data.")


if __name__ == "__main__":
    test_clean_empty_values()
