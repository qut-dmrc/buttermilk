"""Tests for the canonical-once trace-schema change (task buttermilk-1e23cce6).

Locks the dedup + ground-truth-home contract that the design + strategic review required:

- ground_truth (and its hash) have a canonical home on the `record` (Record.model_dump),
  so the BigQuery `record` column carries the true label instead of it surviving only in
  the `inputs` blob.
- The trace `metadata` no longer carries an `input` copy of the record's metadata
  (the `record` column is the single canonical copy; canonical-once).
- The trace-only resolved-inputs projection never re-embeds record-derived artifacts or the
  context message list (those have their own canonical homes: the `record` column and the
  `messages` column). Genuine render variables are retained.

These run against real buttermilk code (no internal mocking).
"""

from buttermilk._core.llm_core import LLMCore
from buttermilk._core.processor_core import ObservabilityMixin
from buttermilk._core.types import Record


class TestGroundTruthCanonicalHome:
    """ground_truth value lives on the record; its HASH is a tuple in record_hashes[].

    Convention (tja-858fc5aa, buttermilk-855514df): hashes are tuple-lists, NOT per-field
    scalar columns. So there is no ground_truth_hash dump key — the hash is an entry in
    record_hashes with content_type='ground_truth'.
    """

    def test_ground_truth_value_present_hash_is_a_tuple_not_a_column(self):
        rec = Record(content="some article text", ground_truth={"violating": True})
        dumped = rec.model_dump()

        # The true label VALUE is present (canonical home on the record → BQ record column).
        assert dumped.get("ground_truth") == {"violating": True}
        # There is NO scalar ground_truth_hash dump key (anti-pattern we removed).
        assert "ground_truth_hash" not in dumped
        # Its hash lives as a tuple in record_hashes[].
        gt_tuples = [h for h in dumped["record_hashes"] if h["content_type"] == "ground_truth"]
        assert len(gt_tuples) == 1
        assert len(gt_tuples[0]["hash"]) == 64  # SHA256

    def test_record_hashes_includes_whole_record_markdown_tuple(self):
        rec = Record(content="some article text", ground_truth={"violating": True})
        dumped = rec.model_dump()
        md = [h for h in dumped["record_hashes"] if h["content_type"] == "record_markdown"]
        assert len(md) == 1 and len(md[0]["hash"]) == 64

    def test_no_ground_truth_yields_no_ground_truth_tuple(self):
        rec = Record(content="x")
        dumped = rec.model_dump()
        # No ground truth → no ground_truth tuple, no crash, no scalar hash key.
        assert "ground_truth_hash" not in dumped
        assert not [h for h in dumped["record_hashes"] if h["content_type"] == "ground_truth"]


class TestTraceMetadataNoRecordCopy:
    """`metadata.input` (byte-identical copy of record.metadata) must be gone."""

    def test_build_trace_metadata_omits_record_metadata(self):
        rec = Record(content="x", metadata={"history": [{"step": "judge"}], "k": "v"})
        mixin = ObservabilityMixin()

        md = mixin._build_trace_metadata(rec, duration_ms=12.5, extra_metadata={"hashes": {"inputs": []}})

        assert "input" not in md, "metadata.input duplicates the canonical `record` column"
        assert md["duration_ms"] == 12.5
        assert md["hashes"] == {"inputs": []}


class TestResolvedInputsProjection:
    """The persisted trace `inputs` holds only render params, no re-embedded artifacts."""

    def _core(self) -> LLMCore:
        return LLMCore(model="test-model", template="judge", template_vars={"criteria": "be fair"})

    def test_strips_record_derived_and_context(self):
        core = self._core()
        resolved = core._build_resolved_inputs(
            {
                "criteria": "be fair",
                "instructions": "judge once",
                # record-derived / canonical-elsewhere — must be dropped:
                "content": "the whole article again",
                "metadata": {"history": [1, 2, 3]},
                "record": {"record_id": "r1"},
                "records": [{"record_id": "r1"}],
                "context": [{"role": "user", "content": "prior turn"}],
            }
        )

        # Genuine render variables are retained (and remain covered by input_hashes).
        assert resolved["criteria"] == "be fair"
        assert resolved["instructions"] == "judge once"
        # Artifacts with a canonical home elsewhere are not re-embedded here.
        for dropped in ("content", "metadata", "record", "records", "context"):
            assert dropped not in resolved, f"{dropped} must not be persisted in trace inputs"

    def test_merges_static_template_vars(self):
        core = self._core()  # template_vars={"criteria": "be fair"}
        resolved = core._build_resolved_inputs({"instructions": "go"})
        # Static template_vars merge in; nothing record-derived present to strip.
        assert resolved == {"criteria": "be fair", "instructions": "go"}
