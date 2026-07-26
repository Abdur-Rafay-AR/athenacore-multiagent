"""Embeddings without a dependency.

A real sentence-transformer is better, and you should plug one in for serious
work (see :class:`CallableEmbedder`). But a project that only works after a
600 MB download is a project most people never run, so the default is
:class:`HashingEmbedder`: a deterministic, offline, pure-stdlib text embedder.

How it works — it is the classic hashing trick over character n-grams plus word
unigrams and bigrams:

1. Normalise the text (casefold, collapse whitespace, strip punctuation).
2. Emit features: word unigrams, word bigrams, and character 4-grams inside
   words. Character n-grams are what give it robustness to typos, plurals and
   morphology without a stemmer.
3. Hash each feature to a bucket with a stable digest (``blake2b``, so vectors
   are reproducible across processes and Python versions — unlike ``hash()``).
4. Weight by sub-linear term frequency, then L2-normalise.

Cosine similarity over those vectors is a genuine lexical-overlap signal: it will
not capture "car" ≈ "automobile" the way a neural model does, but it captures
paraphrase, inflection and partial overlap far better than keyword matching, and
it is the *second* signal in a hybrid ranker whose first signal is already BM25.
"""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from hashlib import blake2b

_WORD_RE = re.compile(r"[a-z0-9']+")

# Extremely common words carry almost no retrieval signal and would otherwise
# dominate the character n-gram features.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "it's",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "very",
        "can",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "us",
        "our",
        "your",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "too",
        "s",
        "t",
        "just",
        "don",
        "now",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
    ]
)


class Embedder(ABC):
    """Turns text into a fixed-size L2-normalised vector."""

    dims: int

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """Batch hook. Network-backed embedders should override this."""
        return [self.embed(t) for t in texts]

    @property
    def name(self) -> str:
        return type(self).__name__


class HashingEmbedder(Embedder):
    """Deterministic offline embedder (see module docstring)."""

    def __init__(self, dims: int = 256, *, char_ngram: int = 4) -> None:
        if dims < 16:
            raise ValueError("dims must be >= 16")
        self.dims = dims
        self.char_ngram = char_ngram

    def _bucket(self, feature: str) -> tuple[int, int]:
        """Map a feature to ``(index, sign)``.

        The sign bit is the standard signed-hashing trick: it makes collisions
        cancel out on average instead of always inflating a bucket.
        """
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dims, 1 if (value >> 63) & 1 else -1

    def _features(self, text: str) -> dict[str, float]:
        words = _WORD_RE.findall(text.lower())
        counts: dict[str, float] = {}

        def bump(feature: str, weight: float = 1.0) -> None:
            counts[feature] = counts.get(feature, 0.0) + weight

        content_words = [w for w in words if w not in _STOPWORDS]
        for word in content_words:
            bump(f"w:{word}", 1.0)
            if len(word) > self.char_ngram:
                padded = f"^{word}$"
                for i in range(len(padded) - self.char_ngram + 1):
                    bump(f"c:{padded[i : i + self.char_ngram]}", 0.5)
        for left, right in itertools.pairwise(content_words):
            bump(f"b:{left}_{right}", 0.8)
        return counts

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        if not text or not text.strip():
            return vector
        for feature, count in self._features(text).items():
            index, sign = self._bucket(feature)
            # Sub-linear TF: a term appearing 20 times is not 20x as informative.
            vector[index] += sign * (1.0 + math.log(count)) if count > 1 else sign * count
        return l2_normalise(vector)


class CallableEmbedder(Embedder):
    """Adapter for any external embedding function.

    Lets a caller swap in a real model without AthenaCore depending on one::

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = CallableEmbedder(lambda ts: model.encode(list(ts)).tolist(), dims=384)
    """

    def __init__(
        self,
        fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        *,
        dims: int,
        name: str = "callable",
    ) -> None:
        self.fn = fn
        self.dims = dims
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        vectors = self.fn(batch)
        out = [l2_normalise([float(v) for v in vec]) for vec in vectors]
        if any(len(v) != self.dims for v in out):
            raise ValueError(f"{self._name} returned vectors that are not {self.dims}-dimensional")
        return out


def l2_normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to ``[-1, 1]``.

    Assumes but does not require normalised inputs; mismatched lengths compare
    over the shared prefix so a dimension change never crashes recall.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def similarity_to_unit(value: float) -> float:
    """Map cosine ``[-1, 1]`` onto ``[0, 1]`` so it can be blended with other signals."""
    return (value + 1.0) / 2.0
