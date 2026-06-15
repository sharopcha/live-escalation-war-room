"""
Knowledge Agent — RAG over the demo KB.

Two retriever backends, selected automatically at startup:
  • EmbeddingRetriever  — OpenAI text-embedding-3-small (used when OPENAI_API_KEY is set)
  • TfIdfRetriever      — pure-stdlib fallback, no API key required

Band SDK: standalone Python (no framework adapter needed)
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Protocol

from shared.band_agent import BaseBandAgent
from shared.band_client import BandClient
from shared.models import BandMessage, KnowledgeResult

logger = logging.getLogger("agent.knowledge")

KB_DIR = Path(__file__).parent.parent.parent / "kb" / "sample"
CHUNK_SIZE = 400       # characters per chunk
OVERLAP = 80
TOP_K = 3


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

class Chunk(NamedTuple):
    text: str
    source: str


class Retriever(Protocol):
    def load(self, kb_dir: Path) -> None: ...
    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]: ...
    def normalize_score(self, score: float) -> float: ...


# ---------------------------------------------------------------------------
# Shared chunk loader (used by both retrievers)
# ---------------------------------------------------------------------------

def _load_chunks(kb_dir: Path) -> list[Chunk]:
    if not kb_dir.exists():
        logger.warning("KB dir %s not found", kb_dir)
        return []
    docs = list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt"))
    chunks: list[Chunk] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for piece in _chunk(text, CHUNK_SIZE, OVERLAP):
            chunks.append(Chunk(text=piece, source=path.name))
    logger.info("Loaded %d chunks from %d KB files", len(chunks), len(docs))
    return chunks


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retriever (stdlib only — no API key required)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0) * v for t, v in b.items())
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


class TfIdfRetriever:
    """Lexical TF-IDF retriever. Fast, offline, no dependencies.
    Limitation: fails on synonym/paraphrase queries."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}
        self._tf_vecs: list[dict[str, float]] = []

    def load(self, kb_dir: Path) -> None:
        self._chunks = _load_chunks(kb_dir)
        self._build_index()

    def _build_index(self) -> None:
        df: dict[str, int] = defaultdict(int)
        tf_raw: list[dict[str, int]] = []
        for chunk in self._chunks:
            tokens = _tokenise(chunk.text)
            freq: dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            tf_raw.append(dict(freq))
            for t in set(tokens):
                df[t] += 1
        n = len(self._chunks)
        self._idf = {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}
        self._tf_vecs = [
            {t: cnt / max(sum(fr.values()), 1) * self._idf.get(t, 0)
             for t, cnt in fr.items()}
            for fr in tf_raw
        ]

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not query.strip():
            return []
        q_freq: dict[str, int] = defaultdict(int)
        for t in _tokenise(query):
            q_freq[t] += 1
        total = max(sum(q_freq.values()), 1)
        q_vec = {t: (cnt / total) * self._idf.get(t, 0) for t, cnt in q_freq.items()}
        scores = [_cosine_sparse(q_vec, v) for v in self._tf_vecs]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._chunks[i], s) for i, s in ranked[:top_k] if s > 0]

    def normalize_score(self, score: float) -> float:
        """TF-IDF cosine scores sit in ~0–0.33; scale to 0–1."""
        return min(score * 3, 1.0)


# ---------------------------------------------------------------------------
# Embedding retriever (OpenAI text-embedding-3-small)
# ---------------------------------------------------------------------------

def _cosine_dense(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


class EmbeddingRetriever:
    """Semantic retriever using OpenAI text-embedding-3-small.
    Handles synonyms and paraphrases that TF-IDF misses.
    Requires OPENAI_API_KEY."""

    MODEL = "text-embedding-3-small"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def load(self, kb_dir: Path) -> None:
        from openai import OpenAI
        self._chunks = _load_chunks(kb_dir)
        if not self._chunks:
            return
        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.embeddings.create(
            model=self.MODEL,
            input=[c.text for c in self._chunks],
        )
        self._embeddings = [e.embedding for e in resp.data]
        logger.info(
            "Embedded %d chunks with %s (dim=%d)",
            len(self._chunks), self.MODEL, len(self._embeddings[0]),
        )

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not self._chunks or not query.strip():
            return []
        from openai import OpenAI
        resp = OpenAI().embeddings.create(model=self.MODEL, input=[query])
        q_vec = resp.data[0].embedding
        scores = [_cosine_dense(q_vec, e) for e in self._embeddings]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._chunks[i], s) for i, s in ranked[:top_k] if s > 0]

    def normalize_score(self, score: float) -> float:
        """Embedding cosine scores are already in ~0.5–1.0 for relevant hits."""
        return min(score, 1.0)


# ---------------------------------------------------------------------------
# LLM answer synthesis (optional — falls back to top passage if no key)
# ---------------------------------------------------------------------------

def _synthesise_answer(query: str, passages: list[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # No LLM — return the top passage directly
        return passages[0] if passages else "No relevant information found in the knowledge base."

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=300)
        context = "\n\n---\n\n".join(passages)
        response = llm.invoke([
            SystemMessage(content=(
                "You are a helpful knowledge agent. Answer the question using ONLY "
                "the provided knowledge base passages. Be concise (2-3 sentences max)."
            )),
            HumanMessage(content=f"Question: {query}\n\nPassages:\n{context}"),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM synthesis failed: %s — using top passage", exc)
        return passages[0] if passages else "No relevant information found."


# ---------------------------------------------------------------------------
# Band agent
# ---------------------------------------------------------------------------

class KnowledgeAgent(BaseBandAgent):
    """RAG knowledge agent. Uses EmbeddingRetriever when OPENAI_API_KEY is
    set, otherwise falls back to TfIdfRetriever."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if os.getenv("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY found — using EmbeddingRetriever")
            self._retriever: Retriever = EmbeddingRetriever()
        else:
            logger.info("No OPENAI_API_KEY — falling back to TfIdfRetriever")
            self._retriever = TfIdfRetriever()
        self._retriever.load(KB_DIR)

    async def handle_message(self, message: BandMessage, client: BandClient) -> None:
        # Extract escalation context
        esc_id = _extract_esc_id(message.text)
        issue = _extract_issue(message.text)
        if not esc_id or not issue:
            logger.debug("Could not parse escalation context, skipping")
            return

        logger.info("Knowledge lookup for %s: %s", esc_id, issue[:80])

        hits = self._retriever.search(issue)
        passages = [c.text for c, _ in hits]
        top_score = hits[0][1] if hits else 0.0
        top_source = hits[0][0].source if hits else "unknown"

        answer = _synthesise_answer(issue, passages)

        result = KnowledgeResult(
            escalation_id=esc_id,
            answer=answer,
            source=top_source,
            confidence=round(self._retriever.normalize_score(top_score), 3),
            supporting_passages=passages[:2],
        )

        bridge_id = os.getenv("BRIDGE_AGENT_ID", "")
        await client.send_message(
            message.chat_id,
            f"@Bridge {result.model_dump_json()}",
            mention_ids=[bridge_id] if bridge_id else [],
        )
        logger.info("Knowledge result sent for %s (confidence=%.2f)", esc_id, result.confidence)


def _extract_esc_id(text: str) -> str | None:
    m = re.search(r"(esc_\w+)", text)
    return m.group(1) if m else None


def _extract_issue(text: str) -> str | None:
    m = re.search(r"\*\*Issue:\*\*\s*(.+)", text)
    if m:
        return m.group(1).strip()
    # Fallback: use the full message text as query
    return text[:300]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = KnowledgeAgent(
        agent_key=os.environ["KNOWLEDGE_AGENT_KEY"],
        base_url=os.getenv("BAND_BASE_URL", "https://app.band.ai"),
        agent_name="knowledge",
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
