import pytest
from langchain_core.prompts import ChatPromptTemplate

from buttermilk.utils.templating import make_messages


def test_make_messages_with_valid_input():
    # Test with valid input
    template = "extraneous prompty data\n\n--# system:\n{{key1}} \nUser: {{key2}}"
    expected_output = (
        "Expected template output"  # Replace with the actual expected output
    )
    input_data = {"key1": expected_output}
    messages = make_messages(template)
    assert len(messages) == 2
    assert messages[0][0] == "system"
    assert messages[1][0] == "User"
    assert messages[1][1] == "{{key2}}"

    compiled_template = ChatPromptTemplate.from_messages(
        messages,
        template_format="jinja2",
    ).format(input=input_data, format="jinja2")

    assert "{{key2}}" in compiled_template
    assert expected_output in compiled_template


def test_make_messages_with_invalid_input():
    # Test with invalid input
    input_data = "invalid_input"
    with pytest.raises(TypeError):
        make_messages(input_data)
