import copy
from buttermilk.utils.utils import scrub_keys

def test_scrub_keys_basic():
    """Test basic dictionary scrubbing."""
    data = {
        "name": "John Doe",
        "api_key": "secret-123",
        "token": "token-456",
        "password": "password-789",
        "secret": "my-secret",
        "credential": "my-credential",
        "public_info": "nothing special"
    }
    expected = {
        "name": "John Doe",
        "public_info": "nothing special"
    }
    assert scrub_keys(data) == expected

def test_scrub_keys_nested():
    """Test nested dictionary scrubbing."""
    data = {
        "user": {
            "id": 1,
            "api_key": "secret-123",
            "profile": {
                "bio": "hello",
                "password": "hidden"
            }
        },
        "items": [
            {"id": 1, "token": "abc"},
            {"id": 2, "name": "item2"}
        ],
        "meta": {
            "secret_field": "val"
        }
    }
    expected = {
        "user": {
            "id": 1,
            "profile": {
                "bio": "hello"
            }
        },
        "items": [
            {"id": 1},
            {"id": 2, "name": "item2"}
        ],
        "meta": {}
    }
    assert scrub_keys(data) == expected

def test_scrub_keys_casing_and_partial():
    """Test case insensitivity and partial matching."""
    data = {
        "API_KEY": "val1",
        "AccessToken": "val2",
        "USER_PASSWORD": "val3",
        "SharedSecret": "val4",
        "userCredentials": "val5",
        "normal_key_is_not_scrubbed": "wait"
    }
    # "normal_key_is_not_scrubbed" contains "key" so it SHOULD be scrubbed
    expected = {}
    assert scrub_keys(data) == expected

    data2 = {"safe": "value"}
    assert scrub_keys(data2) == {"safe": "value"}

def test_scrub_keys_non_collection():
    """Test that it handles non-collection types correctly."""
    assert scrub_keys("some string") == "some string"
    assert scrub_keys(123) == 123
    assert scrub_keys(None) is None

def test_scrub_keys_non_mutating():
    """Test that it does not modify the original data."""
    data = {
        "name": "John",
        "api_key": "secret"
    }
    data_copy = copy.deepcopy(data)
    result = scrub_keys(data)

    assert data == data_copy
    assert result == {"name": "John"}
    assert result is not data

def test_scrub_keys_empty():
    """Test empty collections."""
    assert scrub_keys({}) == {}
    assert scrub_keys([]) == []

def test_scrub_keys_list_root():
    """Test with a list as the root object."""
    data = [
        {"name": "a", "key": "k1"},
        {"name": "b", "token": "t1"}
    ]
    expected = [
        {"name": "a"},
        {"name": "b"}
    ]
    assert scrub_keys(data) == expected

def test_scrub_keys_mixed_types_in_list():
    """Test list with mixed types."""
    data = ["string", 123, {"key": "val"}, [1, {"password": "pwd"}]]
    expected = ["string", 123, {}, [1, {}]]
    assert scrub_keys(data) == expected
