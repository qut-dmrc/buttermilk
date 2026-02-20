"""Bash command processor for pipeline - executes external tools on records.

This module provides processors that execute bash commands on files,
enabling integration of command-line tools (like pdftotext, imagemagick, etc.)
into the data processing pipeline.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import Record


class BashProcessor(BaseModel):
    """Execute bash commands on record files in the pipeline.

    This processor enables using external command-line tools by:
    1. Taking a Record with a file_path
    2. Executing a bash command on that file
    3. Capturing stdout or reading an output file
    4. Updating the record with the results

    The command can use placeholders:
    - {input_file}: Path to the record's file
    - {output_file}: Path to write output (if specified)

    Examples:
        # Count lines in a file
        BashProcessor(command="wc -l {input_file}", output_field="line_count")

        # Convert PDF to text
        BashProcessor(
            command="pdftotext {input_file} {output_file}",
            output_file="/tmp/output.txt",
            read_output_file=True
        )

        # Get file metadata
        BashProcessor(command="file {input_file}", output_field="file_type")
    """

    model_config = {"arbitrary_types_allowed": True}

    command: str = Field(
        ...,
        description="Bash command to execute. Use {input_file} and {output_file} placeholders",
    )
    output_field: str = Field(default="content", description="Record field to update with command output")
    output_file: str | None = Field(
        default=None,
        description="Path for output file (replaces {output_file} placeholder)",
    )
    read_output_file: bool = Field(default=False, description="Read output from file instead of stdout")
    update_file_path: bool = Field(default=False, description="Update record.file_path to point to output_file")
    timeout_seconds: int = Field(default=300, description="Command timeout in seconds")
    shell: bool = Field(default=True, description="Execute command through shell")

    async def process(
        self,
        record: Record,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        component_name: str = "BashProcessor",
        cancellation_token: Any | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Record, None]:
        """Execute bash command on record's file.

        Args:
            record: Record with file_path to process

        Yields:
            Record: Updated record with command output

        Raises:
            FileNotFoundError: If input file doesn't exist
            ProcessingError: If command fails or times out
        """
        # Verify input file exists
        input_path = Path(record.file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Prepare command with placeholders
        command = self.command
        command = command.replace("{input_file}", str(input_path))

        if self.output_file:
            command = command.replace("{output_file}", self.output_file)

        logger.debug(
            f"Executing bash command for record {record.record_id}",
            command=command,
            input_file=str(input_path),
        )

        # Execute command
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.timeout_seconds,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ProcessingError(
                    f"Bash command failed for {record.record_id}: {error_msg}\nCommand: {command}\nReturn code: {result.returncode}"
                )

        except asyncio.TimeoutError:
            raise ProcessingError(f"Bash command timed out after {self.timeout_seconds}s for {record.record_id}\nCommand: {command}")
        except FileNotFoundError as e:
            # Command not found (e.g., pdftotext not installed)
            raise ProcessingError(f"Command not found: {str(e)}\nCommand: {command}\nMake sure the required tool is installed.")
        except Exception as e:
            raise ProcessingError(f"Error executing bash command for {record.record_id}: {e}\nCommand: {command}")

        # Get output
        if self.read_output_file:
            if not self.output_file:
                raise ProcessingError("read_output_file=True but no output_file specified")

            output_path = Path(self.output_file)
            if not output_path.exists():
                raise ProcessingError(f"Output file not created by command: {output_path}\nCommand: {command}")

            output_content = output_path.read_text(encoding="utf-8", errors="replace")
        else:
            output_content = stdout.decode("utf-8", errors="replace").strip()

        # Update record - use model_copy with update dict (Record is frozen)
        updates = {self.output_field: output_content}

        # Optionally update file_path to point to output file
        if self.update_file_path and self.output_file:
            updates["file_path"] = self.output_file

        updated_record = record.model_copy(update=updates)

        logger.debug(
            f"Bash command completed for {record.record_id}",
            output_length=len(output_content),
            output_field=self.output_field,
        )

        yield updated_record


class PDFToTextProcessor(BashProcessor):
    """Specialized processor for extracting text from PDFs using pdftotext.

    This is a convenience wrapper around BashProcessor configured for pdftotext.
    It uses the Poppler library's pdftotext tool which is faster and more reliable
    than pdfminer for many PDFs.

    Installation:
        Ubuntu/Debian: sudo apt-get install poppler-utils
        Mac: brew install poppler
        Fedora: sudo dnf install poppler-utils

    Examples:
        processor = PDFToTextProcessor()
        async for result in processor.process(pdf_record):
            print(result.content)  # Extracted text
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize PDFToTextProcessor with pdftotext configuration.

        Args:
            **kwargs: Additional arguments to pass to BashProcessor
        """
        # Set defaults for pdftotext
        # Use full path to avoid PATH issues in subprocess environments
        defaults = {
            "command": "/usr/bin/pdftotext -layout -nopgbrk {input_file} -",
            "output_field": "content",
            "read_output_file": False,  # pdftotext outputs to stdout with '-'
            "timeout_seconds": 300,
        }

        # Merge with user overrides
        config = {**defaults, **kwargs}

        super().__init__(**config)

    async def process(
        self,
        record: Record,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        component_name: str = "PDFToTextProcessor",
        cancellation_token: Any | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Record, None]:
        """Extract text from PDF using pdftotext.

        Skips extraction if content doesn't look like PDF metadata placeholder
        (meaning fulltext already exists from Zotero API).

        Args:
            record: Record with PDF file_path

        Yields:
            Record: Record with extracted text in content field

        Raises:
            ProcessingError: If pdftotext not installed or extraction fails
        """
        # Skip if content exists and is NOT a PDF placeholder
        # PDF placeholders look like: "[PDF Document: filename.pdf, Size: 123 bytes, Path: /path]"
        if record.content and not (isinstance(record.content, str) and record.content.startswith("[PDF Document:")):
            # QUALITY GATE 3: Validate cached fulltext before skipping extraction
            # This ensures corrupt fulltext from previous runs doesn't bypass validation
            from buttermilk.utils.text_quality import detect_text_corruption

            corruption_result = detect_text_corruption(record.content)
            if corruption_result["is_corrupted"]:
                # Fail loudly - corrupt cached fulltext must be reprocessed
                raise ProcessingError(
                    f"Cached fulltext for {record.record_id} is corrupt: "
                    f"{corruption_result['corruption_percentage']:.1f}% corruption, "
                    f"CID count: {corruption_result['cid_count']}, "
                    f"newline ratio: {corruption_result['newline_ratio']:.1f}%, "
                    f"avg line length: {corruption_result['avg_line_length']:.1f}. "
                    f"Cannot skip PDF extraction - fulltext must be regenerated from PDF."
                )

            logger.debug(
                f"Skipping PDF extraction for {record.record_id} - fulltext already exists and is valid "
                f"({len(record.content)} chars, corruption: {corruption_result['corruption_percentage']:.1f}%)"
            )
            yield record
            return

        try:
            async for result in super().process(
                record,
                processor_stage=processor_stage,
                parent_trace_id=parent_trace_id,
                component_name=component_name,
                cancellation_token=cancellation_token,
                **kwargs,
            ):
                yield result
        except ProcessingError as e:
            # Add helpful context for pdftotext-specific errors
            if "not found" in str(e).lower() and "pdftotext" in str(e).lower():
                raise ProcessingError(
                    f"pdftotext not found. Please install poppler-utils:\n"
                    f"  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    f"  Mac: brew install poppler\n"
                    f"  Fedora: sudo dnf install poppler-utils\n\n"
                    f"Original error: {e}"
                )
            raise
