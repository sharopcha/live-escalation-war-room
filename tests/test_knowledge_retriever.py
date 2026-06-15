"""
tests/test_knowledge_retriever.py

Golden-query tests for both retriever backends in agents/knowledge/main.py.

Run all tests:
    pytest tests/test_knowledge_retriever.py -v

Skip embedding tests (no API key / offline):
    pytest tests/test_knowledge_retriever.py -v -m "not embedding"

There are two kinds of assertions:
  - Source routing  — does the right KB *file* win?
  - Content quality — does the top chunk actually *contain* the answer?

Known failures in the TF-IDF tests are marked xfail with an explanation
of the root cause and the recommended fix, so they document bugs rather
than hide them.  The same queries are tested against EmbeddingRetriever
and are expected to pass there.
"""
from __future__ import annotations

import os

import pytest

from agents.knowledge.main import (
    TfIdfRetriever,
    EmbeddingRetriever,
    KB_DIR,
    _extract_esc_id,
    _extract_issue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever() -> TfIdfRetriever:
    r = TfIdfRetriever()
    r.load(KB_DIR)
    return r


def top_hit(retriever: TfIdfRetriever, query: str):
    """Return (Chunk, score) for the best result, or fail if empty."""
    hits = retriever.search(query)
    assert hits, f"No results at all for query: {query!r}"
    return hits[0]


# ---------------------------------------------------------------------------
# Source routing — does the right KB file win?
# ---------------------------------------------------------------------------

class TestSourceRouting:
    """The retriever must direct billing queries to telecom_billing.md
    and vehicle queries to rar_faq.md.  These should all pass with the
    current fixed-size chunking."""

    @pytest.mark.parametrize("query", [
        "Billed twice in March, wants refund",
        "Customer was charged after cancellation",
        "Internet was down for 6 hours",
        "Customer cannot afford the bill this month",
        "What is the hardship refund process",
        "Duplicate charge on invoice",
    ])
    def test_billing_query_routes_to_billing_kb(self, retriever, query):
        chunk, _ = top_hit(retriever, query)
        assert chunk.source == "telecom_billing.md", (
            f"Expected telecom_billing.md but got {chunk.source!r} for: {query!r}"
        )

    @pytest.mark.parametrize("query", [
        "How do I register an imported EU car",
        "ITP failed, can I still drive",
        "Lost vehicle registration certificate",
        "vehicle tax refund overpayment",
        "How long does vehicle registration take",
        "What documents are needed to register a vehicle",
    ])
    def test_vehicle_query_routes_to_rar_kb(self, retriever, query):
        chunk, _ = top_hit(retriever, query)
        assert chunk.source == "rar_faq.md", (
            f"Expected rar_faq.md but got {chunk.source!r} for: {query!r}"
        )


# ---------------------------------------------------------------------------
# Content quality — does the top chunk actually contain the answer?
# ---------------------------------------------------------------------------

class TestContentQuality:
    """Stricter than source routing: checks that the returned *text*
    contains the relevant policy, not just that we picked the right file.

    xfail cases document known chunk-boundary bugs — fix them by switching
    to semantic Q&A chunking (split on '**Q:**' boundaries)."""

    def test_double_charge_chunk_contains_policy(self, retriever):
        chunk, _ = top_hit(retriever, "Billed twice, wants refund")
        text = chunk.text.lower()
        assert "double charge" in text or "billed twice" in text or "duplicate" in text

    def test_cancellation_chunk_contains_cancellation_answer(self, retriever):
        chunk, _ = top_hit(retriever, "Customer was charged after cancellation")
        assert "cancel" in chunk.text.lower()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Chunk boundary bleed: 'internet was down for 6 hours' returns a chunk "
            "whose PRIMARY topic is the cancellation Q&A. The outage keyword only "
            "appears near the end due to 80-char overlap from the adjacent chunk. "
            "The LLM therefore receives cancellation noise alongside the outage answer. "
            "Fix: replace fixed-size chunking with Q&A-boundary chunking "
            "(split on '**Q:**' markers) so each chunk has exactly one topic."
        ),
    )
    def test_outage_top_chunk_is_primarily_about_outages(self, retriever):
        """The chunk's *primary* topic should be outages, not cancellation.
        We check that cancellation language does not dominate the first half."""
        chunk, _ = top_hit(retriever, "Internet was down for 6 hours")
        first_half = chunk.text[: len(chunk.text) // 2].lower()
        # If 'cancel' dominates the first half, the chunk is mis-ranked
        assert "cancel" not in first_half, (
            "Top chunk primary topic is cancellation, not outage — chunk boundary bleed."
        )

    def test_hardship_chunk_contains_supervisor_process(self, retriever):
        chunk, _ = top_hit(retriever, "Customer cannot afford the bill this month")
        text = chunk.text.lower()
        assert "hardship" in text or "supervisor" in text

    def test_itp_failure_chunk_contains_roadworthy_answer(self, retriever):
        chunk, _ = top_hit(retriever, "ITP failed, can I still drive")
        text = chunk.text.lower()
        assert "itp" in text or "roadworthy" in text or "drive" in text

    def test_lost_cert_chunk_contains_replacement_steps(self, retriever):
        chunk, _ = top_hit(retriever, "Lost vehicle registration certificate")
        text = chunk.text.lower()
        assert "lost" in text or "replacement" in text or "replace" in text or "police" in text

    def test_vehicle_tax_chunk_contains_refund_policy(self, retriever):
        chunk, _ = top_hit(retriever, "vehicle tax refund overpayment")
        text = chunk.text.lower()
        assert "refund" in text or "overpayment" in text or "tax" in text

    def test_known_defect_chunk_contains_credit_table(self, retriever):
        chunk, _ = top_hit(retriever, "DEF-2024-087 4G congestion credit")
        text = chunk.text.lower()
        assert "def-2024" in text or "congestion" in text or "goodwill" in text


# ---------------------------------------------------------------------------
# Vocabulary / synonym coverage (known TF-IDF limitation)
# ---------------------------------------------------------------------------

class TestVocabularyCoverage:
    """TF-IDF is lexical — it fails when the caller uses different words
    than the KB.  These xfail tests document that boundary so we know
    when an embedding upgrade becomes necessary."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TF-IDF vocabulary mismatch: 'service disruption' scores 0.34 "
            "because the KB uses 'outage', not 'disruption'. "
            "Fix: replace TF-IDF with OpenAI text-embedding-3-small — "
            "the API key is already available in .env."
        ),
    )
    def test_synonym_disruption_finds_outage_chunk(self, retriever):
        chunk, score = top_hit(retriever, "service disruption compensation")
        conf = min(score * 3, 1.0)
        assert conf >= 0.5
        assert "outage" in chunk.text.lower() or "credit" in chunk.text.lower()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Chunk boundary bleed: 'duplicate invoice line' finds a chunk that "
            "mentions 'same invoice line' only because it bleeds in from the "
            "neighbouring chunk. The chunk's primary topic is the outage credit "
            "table, not the double-charge policy. "
            "Fix: Q&A-boundary chunking ensures 'duplicate invoice' maps cleanly "
            "to the double-charge block."
        ),
    )
    def test_synonym_duplicate_invoice_top_chunk_is_double_charge(self, retriever):
        """The top chunk should open with double-charge content, not outage content.
        Currently the chunk starts mid-bullet-list with 'Verified network outage'
        because fixed-size cutting split the billing policy list across two chunks."""
        chunk, _ = top_hit(retriever, "duplicate invoice line")
        opening = chunk.text[:80].lower()
        assert "double charge" in opening or "billed twice" in opening or "duplicate" in opening, (
            f"Chunk opens with unrelated content — chunk boundary bleed. Opening: {opening!r}"
        )


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:

    def test_confidence_always_in_range(self, retriever):
        """min(score * 3, 1.0) must stay in [0, 1] for any query."""
        for query in [
            "billed twice",
            "itp failed",
            "service disruption compensation",
            "xyzzy gibberish",
        ]:
            for chunk, score in retriever.search(query):
                conf = min(score * 3, 1.0)
                assert 0.0 <= conf <= 1.0, f"Out-of-range confidence {conf} for {query!r}"

    def test_exact_kb_terms_yield_high_confidence(self, retriever):
        """Queries using KB vocabulary verbatim should score >= 0.5."""
        _, score = top_hit(retriever, "double charge full refund duplicate amount")
        conf = min(score * 3, 1.0)
        assert conf >= 0.5, f"Expected high confidence for exact KB terms, got {conf}"

    def test_top_k_limits_results(self, retriever):
        hits = retriever.search("refund billing credit", top_k=2)
        assert len(hits) <= 2

    def test_top_k_default_is_three(self, retriever):
        hits = retriever.search("refund")
        assert len(hits) <= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_query_returns_no_results(self, retriever):
        assert retriever.search("") == []

    def test_whitespace_only_returns_no_results(self, retriever):
        assert retriever.search("   ") == []

    def test_gibberish_returns_no_results(self, retriever):
        assert retriever.search("xyzzy frobnicator blarg") == []

    def test_results_are_sorted_by_score_descending(self, retriever):
        hits = retriever.search("refund billing")
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_positive(self, retriever):
        for _, score in retriever.search("refund billing"):
            assert score > 0


# ---------------------------------------------------------------------------
# Embedding retriever tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_retriever() -> EmbeddingRetriever:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping embedding tests")
    r = EmbeddingRetriever()
    r.load(KB_DIR)
    return r


@pytest.mark.embedding
class TestEmbeddingRetriever:
    """Tests that require OPENAI_API_KEY and hit the real embedding API.

    These cover the same queries that TF-IDF xfails on, plus a full
    source-routing sweep to confirm semantic retrieval doesn't regress
    on the cases TF-IDF already handles.

    Skip offline:  pytest -m "not embedding"
    """

    # --- Source routing (same coverage as TF-IDF, must not regress) ---

    @pytest.mark.parametrize("query", [
        "Billed twice in March, wants refund",
        "Customer was charged after cancellation",
        "Internet was down for 6 hours",
        "Customer cannot afford the bill this month",
        "Duplicate charge on invoice",
    ])
    def test_billing_query_routes_to_billing_kb(self, embedding_retriever, query):
        chunk, _ = top_hit(embedding_retriever, query)
        assert chunk.source == "telecom_billing.md", (
            f"Expected telecom_billing.md, got {chunk.source!r} for: {query!r}"
        )

    @pytest.mark.parametrize("query", [
        "How do I register an imported EU car",
        "ITP failed, can I still drive",
        "Lost vehicle registration certificate",
        "vehicle tax refund overpayment",
    ])
    def test_vehicle_query_routes_to_rar_kb(self, embedding_retriever, query):
        chunk, _ = top_hit(embedding_retriever, query)
        assert chunk.source == "rar_faq.md", (
            f"Expected rar_faq.md, got {chunk.source!r} for: {query!r}"
        )

    # --- Synonym queries that TF-IDF xfails on ---

    def test_synonym_disruption_finds_outage_content(self, embedding_retriever):
        """'service disruption' should find the outage credit policy."""
        chunk, score = top_hit(embedding_retriever, "service disruption compensation")
        assert chunk.source == "telecom_billing.md"
        assert "outage" in chunk.text.lower() or "credit" in chunk.text.lower()
        # Embedding scores are already in 0–1; meaningful hits score > 0.5
        assert score > 0.5, f"Low confidence {score:.3f} — retrieval may be wrong"

    def test_synonym_duplicate_invoice_finds_double_charge_chunk(self, embedding_retriever):
        """'duplicate invoice line' should find the double-charge policy."""
        chunk, score = top_hit(embedding_retriever, "duplicate invoice line")
        assert chunk.source == "telecom_billing.md"
        assert (
            "double charge" in chunk.text.lower()
            or "billed twice" in chunk.text.lower()
            or "duplicate" in chunk.text.lower()
        )
        assert score > 0.5

    def test_verbose_paraphrase_finds_cancellation_policy(self, embedding_retriever):
        """Long natural-language paraphrase, no exact KB keywords."""
        chunk, score = top_hit(
            embedding_retriever,
            "my subscription was terminated but I keep getting invoices",
        )
        assert chunk.source == "telecom_billing.md"
        assert "cancel" in chunk.text.lower() or "subscription" in chunk.text.lower()
        assert score > 0.5

    def test_synonym_inspection_failure_finds_itp_content(self, embedding_retriever):
        """'vehicle inspection failure' → ITP failure chunk."""
        chunk, score = top_hit(embedding_retriever, "vehicle inspection failure")
        assert chunk.source == "rar_faq.md"
        assert "itp" in chunk.text.lower() or "inspection" in chunk.text.lower()
        assert score > 0.5

    # --- Confidence / score properties ---

    def test_scores_in_range(self, embedding_retriever):
        for query in ["billing error", "ITP renewal", "service disruption"]:
            for chunk, score in embedding_retriever.search(query):
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for {query!r}"

    def test_normalize_score_clamps_to_one(self, embedding_retriever):
        assert embedding_retriever.normalize_score(0.95) == 0.95
        assert embedding_retriever.normalize_score(1.5) == 1.0

    # --- Edge cases ---

    def test_empty_query_returns_no_results(self, embedding_retriever):
        assert embedding_retriever.search("") == []

    def test_whitespace_only_returns_no_results(self, embedding_retriever):
        assert embedding_retriever.search("   ") == []

    def test_results_sorted_descending(self, embedding_retriever):
        hits = embedding_retriever.search("billing refund policy")
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, embedding_retriever):
        assert len(embedding_retriever.search("refund", top_k=2)) <= 2


# ---------------------------------------------------------------------------
# Message parsing (_extract_esc_id / _extract_issue)
# ---------------------------------------------------------------------------

class TestMessageParsing:
    """Tests for the regex helpers that pull escalation context
    out of incoming Band messages."""

    def test_extract_esc_id_standard(self):
        text = "@knowledge esc_abc123def **Issue:** Customer billed twice"
        assert _extract_esc_id(text) == "esc_abc123def"

    def test_extract_esc_id_hex_suffix(self):
        text = "esc_9f4a2b1c **Issue:** Something"
        assert _extract_esc_id(text) == "esc_9f4a2b1c"

    def test_extract_esc_id_missing_returns_none(self):
        assert _extract_esc_id("no escalation id here") is None

    def test_extract_esc_id_empty_string(self):
        assert _extract_esc_id("") is None

    def test_extract_issue_from_bold_marker(self):
        text = "@knowledge esc_abc123 **Issue:** Customer was billed twice in March"
        assert _extract_issue(text) == "Customer was billed twice in March"

    def test_extract_issue_strips_leading_whitespace(self):
        text = "esc_x **Issue:**   leading spaces"
        assert _extract_issue(text) == "leading spaces"

    def test_extract_issue_fallback_to_full_text(self):
        text = "some message without the bold issue marker"
        assert _extract_issue(text) == text[:300]

    def test_extract_issue_fallback_truncates_at_300(self):
        long_text = "a" * 500
        assert _extract_issue(long_text) == long_text[:300]
