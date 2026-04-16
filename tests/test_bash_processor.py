"""Tests for BashProcessor - executes bash commands on records in pipeline.

This processor enables using external command-line tools (like pdftotext)
in the data processing pipeline.
"""

import pytest

from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import Record


def make_context(record: Record) -> ProcessingContext:
    """Helper to wrap a Record in a ProcessingContext for tests."""
    return ProcessingContext(session_id="test", record=record)


class TestBashProcessorBasics:
    """Test basic BashProcessor functionality."""

    @pytest.mark.anyio
    async def test_bash_processor_executes_command_on_file(self, tmp_path):
        """BashProcessor should execute bash command on input file and update record.

        The processor should:
        1. Take a Record with a file_path
        2. Execute a bash command on that file
        3. Capture the output
        4. Update the record's content with the output
        """
        from buttermilk.processors.bash import BashProcessor

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\nLine 2\nLine 3")

        # Create a record with the file
        record = Record(
            record_id="test_001",
            content="placeholder",  # Will be replaced by command output
            file_path=str(test_file),
            metadata={"title": "Test Document"},
        )

        # Create processor that outputs file content
        processor = BashProcessor(
            command="cat {input_file}",
            output_field="content",
        )

        # Process the record
        results = [r async for r in processor.process(make_context(record))]

        # Verify output
        assert len(results) == 1
        result = results[0]
        # Should contain the file content
        assert "Hello World" in result.content
        assert "Line 2" in result.content
        assert result.record_id == "test_001"

    @pytest.mark.anyio
    async def test_bash_processor_replaces_placeholders(self, tmp_path):
        """BashProcessor should replace {input_file} and {output_file} placeholders."""
        from buttermilk.processors.bash import BashProcessor

        input_file = tmp_path / "input.txt"
        input_file.write_text("test content")

        output_file = tmp_path / "output.txt"

        record = Record(
            record_id="test_002",
            content="placeholder",
            file_path=str(input_file),
            metadata={},
        )

        # Command that copies input to output
        processor = BashProcessor(
            command="cp {input_file} {output_file}",
            output_file=str(output_file),
            output_field="content",
            read_output_file=True,
        )

        results = [r async for r in processor.process(make_context(record))]

        assert len(results) == 1
        assert results[0].content == "test content"
        assert output_file.exists()

    @pytest.mark.anyio
    async def test_bash_processor_handles_command_failure(self, tmp_path):
        """BashProcessor should raise ProcessingError when command fails."""
        from buttermilk.processors.bash import BashProcessor

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        record = Record(
            record_id="test_003",
            content="placeholder",
            file_path=str(test_file),
            metadata={},
        )

        # Command that will fail
        processor = BashProcessor(
            command="false",  # Always exits with error code 1
            output_field="content",
        )

        with pytest.raises(ProcessingError) as exc_info:
            async for _ in processor.process(make_context(record)):
                pass

        assert "command failed" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_bash_processor_captures_stdout(self, tmp_path):
        """BashProcessor should capture stdout from command."""
        from buttermilk.processors.bash import BashProcessor

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        record = Record(
            record_id="test_004",
            content="placeholder",
            file_path=str(test_file),
            metadata={},
        )

        processor = BashProcessor(
            command="echo 'Hello from command'",
            output_field="content",
        )

        results = [r async for r in processor.process(make_context(record))]

        assert len(results) == 1
        assert "Hello from command" in results[0].content


class TestPDFToTextProcessor:
    """Test PDFToTextProcessor - specialized processor for pdftotext."""

    @pytest.mark.anyio
    @pytest.mark.skipif(not __import__("shutil").which("pdftotext"), reason="pdftotext not installed (install poppler-utils)")
    async def test_pdftotext_processor_extracts_text_from_pdf(self, tmp_path):
        """PDFToTextProcessor should use pdftotext to extract text from PDFs.

        This is a convenience class that wraps BashProcessor with pdftotext configuration.
        """
        import urllib.request

        from buttermilk.processors.bash import PDFToTextProcessor

        # Download a test PDF
        test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_file = tmp_path / "test.pdf"
        urllib.request.urlretrieve(test_url, pdf_file)

        record = Record(
            record_id="pdf_001",
            content="[PDF Document: test.pdf, Size: 0 bytes, Path: /tmp/test.pdf]",
            file_path=str(pdf_file),
            metadata={"title": "Dummy PDF"},
        )

        processor = PDFToTextProcessor()

        results = [r async for r in processor.process(make_context(record))]

        assert len(results) == 1
        result = results[0]
        assert len(result.content) > 0
        assert "Dummy PDF file" in result.content or "dummy" in result.content.lower()

    @pytest.mark.anyio
    async def test_pdftotext_processor_raises_on_missing_pdftotext(self, tmp_path, monkeypatch):
        """PDFToTextProcessor should raise helpful error if pdftotext not installed."""
        import asyncio

        from buttermilk.processors.bash import PDFToTextProcessor

        # Mock asyncio.create_subprocess_shell to simulate pdftotext not found

        async def mock_create_subprocess(*args, **kwargs):
            raise FileNotFoundError("pdftotext: command not found")

        monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest")

        record = Record(
            record_id="pdf_002",
            content="[PDF Document: test.pdf, Size: 0 bytes, Path: /tmp/test.pdf]",
            file_path=str(pdf_file),
            metadata={},
        )

        processor = PDFToTextProcessor()

        with pytest.raises(ProcessingError) as exc_info:
            async for _ in processor.process(make_context(record)):
                pass

        error_msg = str(exc_info.value).lower()
        assert "pdftotext" in error_msg
        assert "not found" in error_msg or "not installed" in error_msg or "install poppler" in error_msg

    @pytest.mark.anyio
    async def test_pdftotext_processor_uses_processing_context(self, tmp_path, monkeypatch):
        """PDFToTextProcessor.process() must accept ProcessingContext and forward it to super().

        Regression test for partial migration bug: the method signature was updated to accept
        context: ProcessingContext, but the super().process() call still referenced old positional
        params (parent_trace_id, component_name, cancellation_token, kwargs) that are not in scope,
        causing NameError at runtime for every record whose content is a PDF placeholder.

        This test will FAIL with NameError before the fix and PASS after.
        """
        import asyncio

        from buttermilk.processors.bash import PDFToTextProcessor

        # Mock subprocess so we never touch the filesystem for pdftotext execution —
        # we only need to reach (and survive) the super().process(context) call.
        async def mock_proc_communicate():
            return b"extracted text from pdf", b""

        class MockProcess:
            returncode = 0

            async def communicate(self):
                return b"extracted text from pdf", b""

        async def mock_create_subprocess(*args, **kwargs):
            return MockProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess)

        # Create a real PDF-like file so BashProcessor's file-existence check passes
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        record = Record(
            record_id="pdf_context_001",
            # Content is a PDF placeholder — triggers the super().process() branch
            content="[PDF Document: test.pdf, Size: 21 bytes, Path: /tmp/test.pdf]",
            file_path=str(pdf_file),
            metadata={"title": "Context API test"},
        )

        context = ProcessingContext(
            session_id="test-session-001",
            record=record,
        )

        processor = PDFToTextProcessor()

        # Before the fix this raises: NameError: name 'parent_trace_id' is not defined
        # After the fix it should yield one updated record.
        results = [r async for r in processor.process(context)]

        assert len(results) == 1
        assert results[0].content == "extracted text from pdf"


class TestBashProcessorCaching:
    """Test that BashProcessor properly handles file caching."""

    @pytest.mark.anyio
    async def test_bash_processor_updates_cache_file_metadata(self, tmp_path):
        """BashProcessor should update record metadata to point to output file."""
        from buttermilk.processors.bash import BashProcessor

        input_file = tmp_path / "input.txt"
        input_file.write_text("original content")

        output_file = tmp_path / "output.txt"

        record = Record(
            record_id="cache_001",
            content="placeholder",
            file_path=str(input_file),
            metadata={"source": "test"},
        )

        processor = BashProcessor(
            command="echo 'processed' > {output_file}",
            output_file=str(output_file),
            output_field="content",
            read_output_file=True,
            update_file_path=True,  # Update file_path to point to output
        )

        results = [r async for r in processor.process(make_context(record))]

        assert len(results) == 1
        result = results[0]
        assert result.file_path == str(output_file)
        assert "processed" in result.content


class TestBashProcessorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.anyio
    async def test_bash_processor_handles_missing_input_file(self):
        """BashProcessor should raise error if input file doesn't exist."""
        from buttermilk.processors.bash import BashProcessor

        record = Record(
            record_id="edge_001",
            content="placeholder",
            file_path="/nonexistent/file.txt",
            metadata={},
        )

        processor = BashProcessor(
            command="cat {input_file}",
            output_field="content",
        )

        with pytest.raises((FileNotFoundError, ProcessingError)):
            async for _ in processor.process(make_context(record)):
                pass

    @pytest.mark.anyio
    async def test_bash_processor_handles_empty_stdout(self, tmp_path):
        """BashProcessor should handle commands that produce no output."""
        from buttermilk.processors.bash import BashProcessor

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        record = Record(
            record_id="edge_002",
            content="original",
            file_path=str(test_file),
            metadata={},
        )

        # Command that produces no output
        processor = BashProcessor(
            command="true",  # Exits successfully but outputs nothing
            output_field="content",
        )

        results = [r async for r in processor.process(make_context(record))]

        assert len(results) == 1
        # Content should be empty or remain unchanged depending on implementation
        assert results[0].content is not None
