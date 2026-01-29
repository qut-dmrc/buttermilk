
import json
import pytest
from unittest.mock import MagicMock, patch
from buttermilk.processors.vertex_batch import VertexBatchProcessor
from buttermilk._core.types import Record
from buttermilk._core.vertex_batch import BatchJobManager, BatchRequest
from buttermilk.utils.templating import TemplateRenderResult

def mock_render_template(template, **kwargs):
    # Return the template content itself as "rendered"
    # Return a mocked object that mimics TemplateRenderResult
    mock_result = MagicMock()
    mock_result.rendered = template
    mock_result.template_hash = "mock_hash"
    return mock_result

def test_vertex_batch_template_parsing_fails_to_split():
    """
    Reproduce the issue where VertexBatchProcessor doesn't split System/User sections
    and BatchJobManager doesn't handle them correctly for Gemini.
    """
    # Template with System and User sections
    template_content = """# System:
You are a helpful judge.

# User:
Criteria: {{ criteria }}
"""

    with patch("buttermilk.processors.vertex_batch.render_template") as mock_render:
        # Mock return value to mimic render_template return (rendered, hash)
        
        mock_result = MagicMock()
        mock_result.rendered = template_content # Assume variables filled
        mock_result.template_hash = "mock_hash"
        mock_render.return_value = mock_result

        processor = VertexBatchProcessor(
            model="gemini-2.0-flash",
            template="mock_template_name", # Just a name now
            template_vars={},
            fail_on_unfilled_parameters=False
        )

        record = Record(
            record_id="rec1",
            content="I love pizza",
            metadata={"criteria": "Is this about food?"}
        )

        # Prepare requests
        requests = processor.prepare_batch_requests([record])
        
        # Check the cached criteria content
        criteria_contents = processor.get_criteria_contents()
        criteria_key = requests[0].criteria_key
        rendered = criteria_contents[criteria_key]
        
        # FIXED: It now returns a tuple (system, user)
        assert isinstance(rendered, tuple)
        assert len(rendered) == 2
        system_part, user_part = rendered
        
        # Markers should be gone
        assert "# System:" not in system_part
        assert "# User:" not in user_part
        
        # Content should be split correctly
        assert "You are a helpful judge" in system_part
        assert "Criteria: {{ criteria }}" in user_part
        
        # Now check JSONL generation for Gemini
        manager = BatchJobManager(client=MagicMock())
        jsonl = manager.build_jsonl(requests, model="gemini-2.0-flash", criteria_contents=criteria_contents)
        
        entry = json.loads(jsonl)
        
        # FIXED: Gemini batch request should now have system instruction and user content
        # System instruction
        assert "system_instruction" in entry["request"]
        assert "You are a helpful judge" in entry["request"]["system_instruction"]["parts"][0]["text"]
        
        # User content (criteria + record)
        parts = entry["request"]["contents"][0]["parts"]
        assert len(parts) == 2
        assert "Criteria: {{ criteria }}" in parts[0]["text"]  # Criteria (user part)
        assert "I love pizza" in parts[1]["text"]                   # Record content

        print(f"JSONL: {jsonl}")

def test_vertex_batch_claude_repro_markers():
    """
    Claude handles criteria_contents but keeps the raw markers.
    """
    template_content = """# System:
System instruction.

# User:
User instruction.
"""
    with patch("buttermilk.processors.vertex_batch.render_template") as mock_render:
        mock_result = MagicMock()
        mock_result.rendered = template_content
        mock_result.template_hash = "mock_hash"
        mock_render.return_value = mock_result

        processor = VertexBatchProcessor(
            model="claude-3-5-sonnet",
            template="mock_template",
            template_vars={},
            fail_on_unfilled_parameters=False
        )
        record = Record(record_id="r1", content="content")
        requests = processor.prepare_batch_requests([record])
        criteria_contents = processor.get_criteria_contents()
        
        manager = BatchJobManager(client=MagicMock())
        jsonl = manager.build_jsonl(requests, model="claude-3-5-sonnet", criteria_contents=criteria_contents)
        entry = json.loads(jsonl)
        
        # FIXED: Check system instruction and user content
        assert entry["request"]["system"] == "System instruction."
        
        # User message should have criteria and content
        content_blocks = entry["request"]["messages"][0]["content"]
        assert len(content_blocks) == 2
        
        claude_criteria = content_blocks[0]["text"]
        claude_record = content_blocks[1]["text"]
        
        # Markers should be gone
        assert "# System:" not in claude_criteria
        assert "User instruction." in claude_criteria
        assert "content" in claude_record
