"""Hybrid memory recall.

The whole value of a persistent-memory system is deciding *which* memories to put
in front of a model. This module implements that decision as an explicit,
inspectable pipeline rather than a similarity call:

1. **Candidate gathering** — union of (a) BM25 hits from the full-text index and
   (b) the most recent live entries for the topic. Recency alone is what makes
   the system work with no query at all (e.g. the summarizer).
2. **Scoring** — four signals, each normalised to ``[0, 1]`` across candidates,
   combined with configurable weights: keyword relevance, embedding cosine,
   exponential recency decay, and salience.
3. **Diversification** — Maximal Marginal Relevance drops near-duplicates, which
   matters a lot here because agents re-state each other constantly.
4. **Budgeting** — entries are admitted until the token budget is spent, so a
   prompt can never blow the context window.

Every returned item keeps its per-signal breakdown so the UI can show an operator
exactly why a memory surfaced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from athenacore.config import RetrievalSettings
from athenacore.memory.embeddings import Embedder, cosine, similarity_to_unit
from athenacore.memory.models import Entry, EntryKind, estimate_tokens, utcnow
from athenacore.memory.store import EntryFilter, MemoryStore, ScoredEntry


@dataclass(slots=True)
class RecallResult:
    """What a recall produced, plus enough metadata to debug it."""

    entries: list[ScoredEntry]
    query: str
    topic: str
    candidates_considered: int = 0
    tokens: int = 0
    truncated: bool = False
    """True when the token budget cut the result short."""

    def texts(self) -> list[str]:
        return [item.entry.content for item in self.entries]

    def as_prompt_context(self, *, header: str = "PRIOR MEMORY") -> str:
        """Render recalled memory as a numbered prompt block.

        Oldest-first, because narrative order helps models reason about how a
        topic evolved, and numbering lets an agent cite ``[3]`` in its answer.
        """
        if not self.entries:
            return ""
        ordered = sorted(self.entries, key=lambda s: s.entry.created_at)
        blocks = [item.entry.as_prompt_block(index=i) for i, item in enumerate(ordered, start=1)]
        body = "\n\n".join(blocks)
        return f"### {header} ({len(ordered)} entries, ~{self.tokens} tokens)\n\n{body}"

    def citation_map(self) -> dict[int, str]:
        """``{citation number: entry id}`` matching :meth:`as_prompt_context`."""
        ordered = sorted(self.entries, key=lambda s: s.entry.created_at)
        return {i: item.entry.id for i, item in enumerate(ordered, start=1)}

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "topic": self.topic,
            "tokens": self.tokens,
            "truncated": self.truncated,
            "candidates_considered": self.candidates_considered,
            "entries": [item.to_dict() for item in self.entries],
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


def _normalise(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a signal across candidates.

    Normalising per-recall rather than globally is deliberate: it makes weights
    mean "relative importance among these candidates", which is stable even as a
    topic grows and raw BM25 magnitudes drift.
    """
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if math.isclose(hi, lo):
        return {k: (1.0 if hi > 0 else 0.0) for k in values}
    span = hi - lo
    return {k: (v - lo) / span for k, v in values.items()}


class MemoryRetriever:
    """Scores and selects memories for a prompt.

    Stateless apart from its collaborators, so one instance can be shared across
    threads and agents.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        settings: RetrievalSettings | None = None,
        embedder: Embedder | None = None,
        token_budget: int = 6000,
    ) -> None:
        self.store = store
        self.settings = settings or RetrievalSettings()
        self.settings.validate()
        self.embedder = embedder
        self.token_budget = token_budget

    # -- public API ----------------------------------------------------------

    def recall(
        self,
        topic: str,
        query: str = "",
        *,
        kinds: Sequence[EntryKind] | None = None,
        exclude_ids: Sequence[str] = (),
        max_entries: int | None = None,
        token_budget: int | None = None,
    ) -> RecallResult:
        cfg = self.settings
        limit = max_entries or cfg.max_entries
        budget = token_budget if token_budget is not None else self.token_budget
        excluded = set(exclude_ids)

        candidates, keyword_scores = self._gather(topic, query, kinds)
        candidates = [e for e in candidates if e.id not in excluded]
        if not candidates:
            return RecallResult(entries=[], query=query, topic=topic)

        scored = self._score(candidates, keyword_scores, query)
        selected = self._diversify(scored, limit)
        return self._apply_budget(selected, query, topic, budget, len(candidates))

    def recent(self, topic: str, *, limit: int = 10, kinds: Sequence[EntryKind] | None = None) -> list[Entry]:
        """Plain chronological recall — no scoring. Use when order is the point."""
        entries = self.store.query_entries(
            EntryFilter(
                topic=topic,
                kinds=kinds,
                include_archived=self.settings.include_archived,
                limit=limit,
                newest_first=True,
            )
        )
        return list(reversed(entries))

    def index_pending(self, *, topic: str | None = None, batch: int = 200) -> int:
        """Embed entries that have no vector yet. Returns how many were indexed.

        Called after writes and by ``athenacore reindex``; keeping it incremental
        means a large existing store becomes searchable without a stop-the-world
        migration.
        """
        if self.embedder is None:
            return 0
        pending = self.store.entries_missing_embeddings(topic=topic, limit=batch)
        if not pending:
            return 0
        vectors = self.embedder.embed_many([e.content for e in pending])
        for entry, vector in zip(pending, vectors):
            self.store.set_embedding(entry.id, vector)
        return len(pending)

    # -- pipeline stages -----------------------------------------------------

    def _gather(
        self, topic: str, query: str, kinds: Sequence[EntryKind] | None
    ) -> tuple[list[Entry], dict[str, float]]:
        cfg = self.settings
        pool: dict[str, Entry] = {}
        keyword_scores: dict[str, float] = {}

        if query.strip() and cfg.keyword_weight > 0:
            for entry, relevance in self.store.keyword_search(
                query, topic=topic, limit=cfg.candidate_pool
            ):
                if not cfg.include_archived and entry.archived:
                    continue
                pool[entry.id] = entry
                keyword_scores[entry.id] = relevance

        recent = self.store.query_entries(
            EntryFilter(
                topic=topic,
                kinds=kinds,
                include_archived=cfg.include_archived,
                limit=cfg.candidate_pool,
                newest_first=True,
            )
        )
        for entry in recent:
            pool.setdefault(entry.id, entry)

        if kinds:
            wanted = set(kinds)
            pool = {k: v for k, v in pool.items() if v.kind in wanted}

        return list(pool.values()), keyword_scores

    def _score(
        self, candidates: list[Entry], keyword_scores: dict[str, float], query: str
    ) -> list[ScoredEntry]:
        cfg = self.settings
        now = utcnow()
        half_life = cfg.recency_half_life_hours

        recency_raw: dict[str, float] = {}
        for entry in candidates:
            age_h = max(0.0, (now - entry.created_at).total_seconds() / 3600.0)
            recency_raw[entry.id] = math.pow(0.5, age_h / half_life)

        semantic_raw: dict[str, float] = {}
        if self.embedder is not None and cfg.semantic_weight > 0 and query.strip():
            query_vec = self.embedder.embed(query)
            stored = self.store.get_embeddings([e.id for e in candidates])
            for entry in candidates:
                vector = stored.get(entry.id)
                if vector is None:
                    # Not yet indexed: embed on the fly so a cold store still ranks
                    # sensibly, and persist it so the next recall is cheaper.
                    vector = self.embedder.embed(entry.content)
                    self.store.set_embedding(entry.id, vector)
                semantic_raw[entry.id] = similarity_to_unit(cosine(query_vec, vector))

        kw_norm = _normalise({e.id: keyword_scores.get(e.id, 0.0) for e in candidates})
        sem_norm = _normalise(semantic_raw) if semantic_raw else {}
        rec_norm = _normalise(recency_raw)

        weight_sum = (
            cfg.keyword_weight
            + (cfg.semantic_weight if sem_norm else 0.0)
            + cfg.recency_weight
            + cfg.salience_weight
        ) or 1.0

        results: list[ScoredEntry] = []
        for entry in candidates:
            kw = kw_norm.get(entry.id, 0.0)
            sem = sem_norm.get(entry.id, 0.0)
            rec = rec_norm.get(entry.id, 0.0)
            sal = entry.salience
            total = (
                cfg.keyword_weight * kw
                + (cfg.semantic_weight * sem if sem_norm else 0.0)
                + cfg.recency_weight * rec
                + cfg.salience_weight * sal
            ) / weight_sum
            # Protected kinds are the topic's condensed backbone; a small boost
            # keeps them from being crowded out by a burst of fresh chatter.
            if entry.kind.protected:
                total = min(1.0, total * 1.15)
            if total >= cfg.min_score:
                results.append(
                    ScoredEntry(
                        entry=entry, score=total, keyword=kw, semantic=sem, recency=rec, salience=sal
                    )
                )
        results.sort(key=lambda s: s.score, reverse=True)
        return results

    def _diversify(self, scored: list[ScoredEntry], limit: int) -> list[ScoredEntry]:
        """Maximal Marginal Relevance selection.

        Greedily picks the candidate maximising
        ``λ·relevance − (1−λ)·max_similarity_to_already_picked``. Without this,
        recall returns five paraphrases of the same critique and wastes the budget.
        """
        cfg = self.settings
        if len(scored) <= limit or cfg.mmr_lambda >= 1.0:
            return scored[:limit]

        vectors = self._vectors_for(scored)
        picked: list[ScoredEntry] = []
        remaining = list(scored)
        while remaining and len(picked) < limit:
            best_item = None
            best_value = -math.inf
            for item in remaining:
                penalty = 0.0
                if picked:
                    penalty = max(
                        similarity_to_unit(
                            cosine(vectors[item.entry.id], vectors[chosen.entry.id])
                        )
                        for chosen in picked
                    )
                value = cfg.mmr_lambda * item.score - (1.0 - cfg.mmr_lambda) * penalty
                if value > best_value:
                    best_value, best_item = value, item
            assert best_item is not None
            picked.append(best_item)
            remaining.remove(best_item)
        return picked

    def _vectors_for(self, scored: list[ScoredEntry]) -> dict[str, list[float]]:
        ids = [item.entry.id for item in scored]
        if self.embedder is None:
            # No embedder: fall back to a bag-of-words proxy so MMR still removes
            # obvious duplicates rather than silently doing nothing.
            return {
                item.entry.id: _lexical_signature(item.entry.content) for item in scored
            }
        stored = self.store.get_embeddings(ids)
        out: dict[str, list[float]] = {}
        for item in scored:
            vector = stored.get(item.entry.id)
            if vector is None:
                vector = self.embedder.embed(item.entry.content)
                self.store.set_embedding(item.entry.id, vector)
            out[item.entry.id] = vector
        return out

    def _apply_budget(
        self,
        selected: list[ScoredEntry],
        query: str,
        topic: str,
        budget: int,
        considered: int,
    ) -> RecallResult:
        admitted: list[ScoredEntry] = []
        used = 0
        truncated = False
        for item in selected:
            cost = item.entry.tokens or estimate_tokens(item.entry.content)
            if admitted and used + cost > budget:
                truncated = True
                continue
            admitted.append(item)
            used += cost
        return RecallResult(
            entries=admitted,
            query=query,
            topic=topic,
            candidates_considered=considered,
            tokens=used,
            truncated=truncated,
        )


def _lexical_signature(text: str, dims: int = 128) -> list[float]:
    """Cheap bag-of-words vector for duplicate detection without an embedder."""
    from athenacore.memory.embeddings import HashingEmbedder

    return HashingEmbedder(dims=dims).embed(text)
