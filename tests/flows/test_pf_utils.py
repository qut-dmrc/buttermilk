from buttermilk.utils.pf import test_col_mapping_hydra_to_pf

def test_col_mapping_hydra_to_pf():
    # Test case 1: Empty dictionary
    assert col_mapping_hydra_to_pf({}) == {}

def test_col_mapping_hydra_to_pf_single():
    # Test case 2: Single key-value pair
    input_dict = {"id": "data.id"}
    expected_output = {"id": "${data.id}"}
    assert col_mapping_hydra_to_pf(input_dict) == expected_output

def test_col_mapping_hydra_to_pf_multi():
    # Test case 3: Multiple key-value pairs
    input_dict = {
        "record_id": "data.id",
        "name": "data.name",
        "age": "data.age"
    }
    expected_output = {
        "record_id": "${data.id}",
        "name": "${data.name}",
        "age": "${data.age}"
    }
    assert col_mapping_hydra_to_pf(input_dict) == expected_output

def test_col_mapping_hydra_to_pf_specialchars():
    # Test case 4: Keys with spaces and special characters
    input_dict = {
        "user id": "data.user_id",
        "full name": "data.full_name",
        "email@address": "data.email"
    }
    expected_output = {
        "user id": "${data.user_id}",
        "full name": "${data.full_name}",
        "email@address": "${data.email}"
    }
    assert col_mapping_hydra_to_pf(input_dict) == expected_output

def test_col_mapping_hydra_to_pf_nested():
    # Test case 5: Values with nested attributes
    input_dict = {
        "address": "data.location.address",
        "city": "data.location.city",
        "country": "data.location.country"
    }
    expected_output = {
        "address": "${data.location.address}",
        "city": "${data.location.city}",
        "country": "${data.location.country}"
    }
    assert  c(input_dict) == expected_output