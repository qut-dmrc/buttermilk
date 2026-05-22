"""Text quality and corruption detection module.

Provides reusable functions for detecting text corruption patterns in PDF extractions,
including CID encoding artifacts and language-based corruption detection.

This module provides corruption detection logic to ensure text quality gates
across the codebase, particularly for Zotero fulltext processing.
"""

import re

from langdetect import DetectorFactory, LangDetectException, detect

# Pattern to detect PDF encoding artifacts like (cid:XX)
CID_PATTERN = re.compile(r"\(cid:\d+\)")

# Ensure consistent language detection results
DetectorFactory.seed = 0


def detect_text_corruption(text: str) -> dict:
    """Detect corruption patterns in text content.

    Uses multiple detection methods:
    1. CID pattern matching: Detects (cid:XX) PDF encoding artifacts (threshold: >=20)
    2. Newline ratio: Excessive newlines indicate corruption (>10% is suspect)
    3. Character separation: Single chars on lines indicate garbled text
    4. Language detection: Logged for analysis only (NOT used for corruption detection)

    Args:
        text: Text content to analyze

    Returns:
        dict: Corruption analysis with keys:
            - is_corrupted: bool indicating if text has corruption (ANY signal)
            - corruption_percentage: float from 0-100 based on corruption signals
            - cid_count: int count of (cid:XX) patterns found
            - newline_ratio: float percentage of text that is newlines
            - avg_line_length: float average characters per line
            - detected_language: str language code from langdetect (for logging only)
    """
    if not text or len(text.strip()) == 0:
        # Empty text is considered completely corrupted
        return {
            "is_corrupted": True,
            "corruption_percentage": 100.0,
            "cid_count": 0,
            "newline_ratio": 0.0,
            "avg_line_length": 0.0,
            "detected_language": "unknown",
        }

    total_chars = len(text)

    # SIGNAL 1: Find all (cid:XX) patterns
    cid_matches = CID_PATTERN.findall(text)
    cid_count = len(cid_matches)
    cid_chars = sum(len(match) for match in cid_matches)
    cid_corruption = (cid_chars / total_chars * 100) if total_chars > 0 else 0.0

    # SIGNAL 2: Newline ratio (excessive newlines indicate corruption)
    newline_count = text.count("\n")
    newline_ratio = (newline_count / total_chars * 100) if total_chars > 0 else 0.0

    # SIGNAL 3: Character separation detection
    # If average line length is very short (< 5 chars), likely corrupted
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0

    # Character separation: many lines with < 3 chars indicates corruption
    short_lines = sum(1 for line in non_empty_lines if len(line.strip()) < 3)
    short_line_ratio = (short_lines / len(non_empty_lines) * 100) if non_empty_lines else 0

    # SIGNAL 4: Language detection (for logging only - NOT used for corruption detection)
    # Language detection previously caused false positives with bibliographies/author names
    detected_language = "unknown"

    try:
        detected_language = detect(text)
    except LangDetectException:
        detected_language = "unknown"

    # Calculate composite corruption percentage
    # Weight different signals:
    # - CID patterns: direct indicator (threshold: >= 20 to ignore minor header CIDs)
    # - Newline ratio > 10%: strong indicator
    # - Short lines > 50%: strong indicator
    # - Average line length < 10: moderate indicator
    corruption_signals = []

    # Require minimum 20 CIDs to flag as corruption (ignores citation headers with 1-14 CIDs)
    if cid_count >= 20:
        corruption_signals.append(cid_corruption)
    if newline_ratio > 10.0:
        corruption_signals.append(min(newline_ratio * 2, 50.0))  # Cap at 50%
    if short_line_ratio > 50.0:
        corruption_signals.append(min(short_line_ratio, 50.0))
    if avg_line_length < 10 and avg_line_length > 0:
        corruption_signals.append(20.0)

    # Use maximum corruption signal as overall percentage
    corruption_percentage = max(corruption_signals) if corruption_signals else 0.0

    # Mark as corrupted if ANY strong signal triggers
    # CID threshold: >= 20 to ignore minor header artifacts
    # Language detection removed - was causing false positives with bibliographies/citations
    is_corrupted = cid_count >= 20 or newline_ratio > 10.0 or short_line_ratio > 50.0 or (avg_line_length < 10 and avg_line_length > 0)

    return {
        "is_corrupted": is_corrupted,
        "corruption_percentage": corruption_percentage,
        "cid_count": cid_count,
        "newline_ratio": newline_ratio,
        "avg_line_length": avg_line_length,
        "detected_language": detected_language,
    }


def is_document_corrupt(chunks: list[str], threshold: float = 66.0) -> dict:
    """Determine if a document is corrupt based on chunk corruption rate.

    Analyzes each chunk for corruption patterns and calculates the percentage
    of corrupted chunks. If this percentage exceeds the threshold, the document
    is flagged as corrupt.

    Args:
        chunks: List of text chunks from a single document
        threshold: Corruption percentage threshold (default: 66.0)
                   Document is corrupt if corruption_rate >= threshold

    Returns:
        dict: Document corruption analysis with keys:
            - is_corrupt: bool indicating if document exceeds corruption threshold
            - corruption_rate: float percentage of chunks that are corrupted (0-100)
            - total_chunks: int total number of chunks analyzed
            - corrupted_chunks: int number of chunks marked as corrupted
            - threshold: float threshold value used for decision
            - chunk_details: list of per-chunk corruption results
    """
    # Handle empty document edge case
    if not chunks:
        return {
            "is_corrupt": True,
            "corruption_rate": 100.0,
            "total_chunks": 0,
            "corrupted_chunks": 0,
            "threshold": threshold,
            "chunk_details": [],
        }

    total_chunks = len(chunks)
    corrupted_chunks = 0
    chunk_details = []

    # Analyze each chunk using existing corruption detection
    for chunk in chunks:
        result = detect_text_corruption(chunk)
        chunk_details.append(result)
        if result["is_corrupted"]:
            corrupted_chunks += 1

    # Calculate corruption rate as percentage
    corruption_rate = (corrupted_chunks / total_chunks * 100) if total_chunks > 0 else 0.0

    # Document is corrupt if corruption rate meets or exceeds threshold
    is_corrupt = corruption_rate >= threshold

    return {
        "is_corrupt": is_corrupt,
        "corruption_rate": corruption_rate,
        "total_chunks": total_chunks,
        "corrupted_chunks": corrupted_chunks,
        "threshold": threshold,
        "chunk_details": chunk_details,
    }


def analyze_document_quality(chunks: list[str]) -> dict:
    """Analyze document quality by aggregating corruption across all chunks.

    This performs document-level analysis by checking each chunk and calculating
    overall corruption statistics.

    Args:
        chunks: List of text chunks from a single document

    Returns:
        dict: Document quality analysis with keys:
            - total_chunks: int total number of chunks analyzed
            - corrupted_chunks: int number of chunks marked as corrupted
            - corruption_rate: float percentage of chunks that are corrupted (0-100)
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "corrupted_chunks": 0,
            "corruption_rate": 0.0,
        }

    total_chunks = len(chunks)
    corrupted_chunks = 0

    for chunk in chunks:
        result = detect_text_corruption(chunk)
        if result["is_corrupted"]:
            corrupted_chunks += 1

    corruption_rate = (corrupted_chunks / total_chunks * 100) if total_chunks > 0 else 0.0

    return {
        "total_chunks": total_chunks,
        "corrupted_chunks": corrupted_chunks,
        "corruption_rate": corruption_rate,
    }
