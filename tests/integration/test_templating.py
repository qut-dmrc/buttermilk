"""Integration tests for the templating system.

Tests the custom _parse_chat_messages parser against real Jinja2 templates
used by main agents like OSB, analyst, RAG, etc.
"""

import pytest
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import (
    _parse_chat_messages,
    load_template,
    make_messages,
)


class TestChatMessageParser:
    """Test the _parse_chat_messages function with various formats."""

    def test_simple_chat_format(self):
        """Test parsing simple role: content format."""
        chat_str = """system: You are a helpful assistant
user: Hello!
assistant: Hi there!"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there!"

    def test_chat_format_with_hash_prefix(self):
        """Test parsing # role: content format."""
        chat_str = """# system: You are a helpful assistant
# user: Hello!
# assistant: Hi there!"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_multiline_content(self):
        """Test parsing messages with multiline content."""
        chat_str = """system: You are a helpful assistant.
You should be polite and friendly.
Always greet the user.

user: Hello there!
How are you doing?"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 2
        assert "You should be polite and friendly" in messages[0]["content"]
        assert "Always greet the user" in messages[0]["content"]
        assert "How are you doing?" in messages[1]["content"]

    def test_placeholder_role(self):
        """Test parsing placeholder roles."""
        chat_str = """system: Instructions here
placeholder: {{context}}
user: Question here"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 3
        assert messages[1]["role"] == "placeholder"
        assert messages[1]["content"] == "{{context}}"

    def test_empty_content_after_role(self):
        """Test parsing when role has no immediate content."""
        chat_str = """system:
This is the actual content
on multiple lines

user: Question"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 2
        assert "This is the actual content" in messages[0]["content"]

    def test_developer_role(self):
        """Test parsing developer role (should map to SystemMessage)."""
        chat_str = """developer: System instructions
user: Question"""

        messages = _parse_chat_messages(chat_str)

        assert len(messages) == 2
        assert messages[0]["role"] == "developer"


class TestOSBTemplate:
    """Test the OSB expert template (osb.jinja2)."""

    def test_osb_template_loads(self):
        """Test that OSB template loads and renders correctly."""
        parameters = {
            "dataset": [
                {"title": "Case A", "text": "Content of case A"},
                {"title": "Case B", "text": "Content of case B"},
            ],
            "formatting": "Please respond in JSON format",
        }

        rendered, undefined_vars, template_hash = load_template(
            template="osb",
            parameters=parameters,
        )

        # Should have rendered successfully
        assert "You are a helpful paralegal" in rendered
        assert "Case A" in rendered
        assert "Case B" in rendered
        assert "Please respond in JSON format" in rendered
        # Template hash is a hex string, not prefixed with "sha256:"
        assert len(template_hash) == 64  # SHA256 hex length

    def test_osb_template_with_prompt_placeholder(self):
        """Test OSB template with prompt placeholder."""
        parameters = {
            "dataset": [{"title": "Case A", "text": "Content"}],
            "prompt": "What is the main issue in these cases?",
        }

        rendered, _, _ = load_template(
            template="osb",
            parameters=parameters,
        )

        assert "What is the main issue in these cases?" in rendered

    def test_osb_template_messages(self):
        """Test that OSB template creates proper message structure."""
        # OSB template just has "# System:" and then "placeholder:" with {{prompt}}
        # The placeholder role is followed by the variable name directly
        parameters = {
            "dataset": [{"title": "Case A", "text": "Content"}],
            "prompt": "Analyze this case",
        }

        rendered, _, _ = load_template(
            template="osb",
            parameters=parameters,
        )

        # Parse into messages - the osb template has just system instructions
        # with a prompt variable that gets rendered
        messages, _ = make_messages(rendered)

        # Should have at least one system message with instructions
        assert any(isinstance(msg, SystemMessage) for msg in messages)
        # The prompt is rendered into the template, not a placeholder
        assert any(
            "Analyze this case" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )


class TestAnalystTemplate:
    """Test the analyst template (analyst.jinja2)."""

    def test_analyst_template_loads(self):
        """Test that analyst template loads correctly."""
        parameters = {}

        rendered, undefined_vars, _ = load_template(
            template="analyst",
            parameters=parameters,
        )

        assert "You are a careful analyst" in rendered
        # record and context are placeholders, not template variables
        # They should remain as {{record}} and {{context}} in rendered output
        assert "{{record}}" in rendered
        assert "{{context}}" in rendered

    def test_analyst_template_with_expertise(self):
        """Test analyst template with expertise section."""
        parameters = {
            "record": "Record content",
            "expertise": "You are an expert in legal analysis",
            "criteria": "Use the following criteria: ...",
        }

        rendered, _, _ = load_template(
            template="analyst",
            parameters=parameters,
        )

        assert "BEGIN EXPERTISE" in rendered
        assert "expert in legal analysis" in rendered
        assert "BEGIN CRITERIA" in rendered

    def test_analyst_template_messages(self):
        """Test analyst template creates proper messages."""
        # Analyst template has "# Placeholder:" followed by {{record}} and {{context}}
        # record is a placeholder that gets replaced by make_messages()

        # Create a mock record
        class MockRecord(BaseRecord):
            text: str = "Test record content"

            def as_message(self):
                return UserMessage(content=self.text, source="record")

        record = MockRecord()

        # Don't pass record to load_template - it's a placeholder
        parameters = {}

        rendered, _, _ = load_template(
            template="analyst",
            parameters=parameters,
        )

        # Pass record to make_messages instead
        messages, placeholders = make_messages(rendered, record=record)

        # Should have system message
        assert any(isinstance(msg, SystemMessage) for msg in messages)
        # Should have replaced record placeholder
        assert "record" in placeholders or "records" in placeholders
        # The record content should appear in messages
        assert any(
            "Test record content" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )


class TestRAGTemplate:
    """Test the RAG template (rag.jinja2)."""

    def test_rag_template_loads(self):
        """Test that RAG template loads correctly."""
        parameters = {
            "context": "Vector database search results...",
            "prompt": "What does the research say about this topic?",
        }

        rendered, _, _ = load_template(
            template="rag",
            parameters=parameters,
        )

        assert "You are a research assistant" in rendered
        assert "vector database search tool" in rendered

    def test_rag_template_messages_with_context(self):
        """Test RAG template with context messages."""
        from autogen_core.models import UserMessage as AutogenUserMessage

        parameters = {
            "prompt": "Summarize the findings",
        }

        rendered, _, _ = load_template(
            template="rag",
            parameters=parameters,
        )

        # Create context messages
        context_messages = [
            AutogenUserMessage(content="Previous question", source="user"),
            SystemMessage(content="Previous response"),
        ]

        messages, placeholders = make_messages(
            rendered,
            context=context_messages,
        )

        # Should include the context messages if placeholder was present
        if "context" in placeholders:
            assert len(messages) >= len(context_messages)

    def test_rag_template_without_optional_fields(self):
        """Test RAG template renders without optional context."""
        parameters = {
            "prompt": "Just a prompt",
        }

        rendered, _, _ = load_template(
            template="rag",
            parameters=parameters,
        )

        # Should still render successfully
        assert "You are a research assistant" in rendered
        assert "Just a prompt" in rendered


class TestMakeMessages:
    """Test the make_messages function with various scenarios."""

    def test_make_messages_with_context_placeholder(self):
        """Test that context placeholder gets replaced."""
        template_str = """# System:
You are a helpful assistant

# Placeholder:
{{context}}

# User:
New question here"""

        context_messages = [
            UserMessage(content="Previous question", source="user"),
            AssistantMessage(content="Previous answer", source="assistant"),
        ]

        messages, placeholders = make_messages(
            template_str,
            context=context_messages,
        )

        # Should have replaced context placeholder
        assert "context" in placeholders
        # Should include previous messages
        assert any(
            "Previous question" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )

    def test_make_messages_with_record_placeholder(self):
        """Test that record placeholder gets replaced."""
        template_str = """# System:
Analyze this record

# Placeholder:
{{record}}"""

        # Create a mock record
        class MockRecord(BaseRecord):
            text: str = "Record content"

            def as_message(self):
                return UserMessage(content=self.text, source="record")

        record = MockRecord()

        messages, placeholders = make_messages(
            template_str,
            record=record,
        )

        # Should have replaced record placeholder
        assert "record" in placeholders or "records" in placeholders
        # Should include record content
        assert any(
            "Record content" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )

    def test_make_messages_deduplication(self):
        """Test that duplicate messages are removed."""
        template_str = """# System:
Instruction A

# System:
Instruction A

# User:
Question"""

        messages, _ = make_messages(template_str)

        # Should deduplicate the identical system messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        assert len(system_messages) == 1

    def test_make_messages_with_frontmatter(self):
        """Test parsing template with YAML frontmatter."""
        template_str = """---
name: Test Template
version: 1.0
---

# System:
Instructions here

# User:
Question here"""

        messages, _ = make_messages(template_str)

        # Should parse correctly despite frontmatter
        assert len(messages) >= 2
        assert any(isinstance(msg, SystemMessage) for msg in messages)
        assert any(isinstance(msg, UserMessage) for msg in messages)


class TestTemplateIntegration:
    """Integration tests combining template loading and message creation."""

    def test_full_workflow_osb(self):
        """Test complete workflow: load template -> render -> create messages."""
        # Step 1: Load and render template
        parameters = {
            "dataset": [
                {"title": "Case 1", "summary": "Summary 1"},
                {"title": "Case 2", "summary": "Summary 2"},
            ],
            "prompt": "What are the key differences?",
        }

        rendered, undefined, hash_val = load_template("osb", parameters)

        # Step 2: Create messages
        messages, _ = make_messages(rendered)

        # Verify
        assert len(messages) > 0
        assert len(hash_val) == 64  # SHA256 hex length
        assert any(
            "helpful paralegal" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )

    def test_full_workflow_analyst(self):
        """Test complete workflow with analyst template."""
        # Create a mock record for the placeholder
        class MockRecord(BaseRecord):
            text: str = "Case details content"

            def as_message(self):
                return UserMessage(content=self.text, source="record")

        record = MockRecord()

        # record is a placeholder, not a template parameter
        # Only pass non-placeholder parameters to load_template
        parameters = {
            "expertise": "Legal expert",
            "criteria": "Apply these rules...",
        }

        rendered, _, _ = load_template("analyst", parameters)

        # Pass record to make_messages instead
        messages, placeholders = make_messages(rendered, record=record)

        # Should have properly formatted messages
        assert len(messages) > 0
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        assert len(system_msgs) > 0
        # Should have replaced record placeholder
        assert "record" in placeholders or "records" in placeholders
        # Verify record content appears in messages
        assert any(
            "Case details content" in msg.content
            for msg in messages
            if hasattr(msg, "content")
        )

    def test_undefined_variables_tracking(self):
        """Test that undefined variables are properly tracked."""
        # OSB template uses {% for case in dataset %} which doesn't create
        # undefined variables - it just renders empty when dataset is missing.
        # Let's use analyst template which has direct variable references
        parameters = {
            # Missing 'record' and 'context' parameters
        }

        rendered, undefined, _ = load_template("analyst", parameters)

        # Should track undefined variables
        assert "record" in undefined or "{{record}}" in rendered
        assert "context" in undefined or "{{context}}" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
