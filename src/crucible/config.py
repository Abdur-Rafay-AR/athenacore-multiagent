"""Configuration.

Settings resolve in this order, later winning: dataclass defaults -> ``.env``
file -> process environment -> explicit keyword arguments. No third-party
settings library is used so that the core stays importable with a bare
interpreter.

Every knob is documented in ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from crucible.errors import ConfigurationError

ENV_PREFIX = "CRUCIBLE_"

DEFAULT_DATA_DIR = Path("data")


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """Parse a ``.env`` file into ``os.environ`` and return what was read.

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines and
    single or double quoted values. Unparseable lines are skipped rather than
    raising, because a hand-edited ``.env`` should never break startup.
    """
    p = Path(path)
    parsed: dict[str, str] = {}
    if not p.is_file():
        return parsed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def _defaults(cls: type) -> dict[str, Any]:
    """Field defaults for a ``slots=True`` dataclass.

    Slotted dataclasses replace class-level defaults with slot descriptors, so
    ``cls.model`` is a descriptor rather than ``"ollama:llama3.1"``. The defaults
    have to be read back off the field metadata.
    """
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
    return out


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{name}={raw!r} is not an integer",
            hint="Remove the value to fall back to the default.",
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_PREFIX}{name}={raw!r} is not a number") from exc


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Weights and limits for hybrid memory recall.

    The three signals are combined linearly after each is normalised to ``[0, 1]``
    across the candidate set, then redundant results are dropped with MMR. Tuning
    these is the single highest-leverage thing an operator can do, so they are
    first-class settings rather than call-site literals.
    """

    max_entries: int = 12
    """How many entries a single recall may return."""

    candidate_pool: int = 60
    """How many rows to score before ranking. Larger is slower but more accurate."""

    keyword_weight: float = 1.0
    semantic_weight: float = 1.0
    recency_weight: float = 0.6
    salience_weight: float = 0.4

    recency_half_life_hours: float = 72.0
    """Age at which the recency signal decays to half its value."""

    mmr_lambda: float = 0.7
    """1.0 = pure relevance, 0.0 = pure diversity."""

    min_score: float = 0.0
    """Drop results scoring below this after normalisation."""

    include_archived: bool = False
    """Whether entries superseded by compaction are eligible for recall."""

    def validate(self) -> None:
        if self.max_entries < 1:
            raise ConfigurationError("retrieval.max_entries must be >= 1")
        if self.candidate_pool < self.max_entries:
            raise ConfigurationError("retrieval.candidate_pool must be >= max_entries")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ConfigurationError("retrieval.mmr_lambda must be within [0, 1]")
        if self.recency_half_life_hours <= 0:
            raise ConfigurationError("retrieval.recency_half_life_hours must be > 0")
        total = (
            self.keyword_weight + self.semantic_weight + self.recency_weight + self.salience_weight
        )
        if total <= 0:
            raise ConfigurationError("at least one retrieval weight must be positive")

    @classmethod
    def from_env(cls) -> RetrievalSettings:
        d = _defaults(cls)
        return cls(
            max_entries=_env_int("RECALL_MAX_ENTRIES", d["max_entries"]),
            candidate_pool=_env_int("RECALL_CANDIDATES", d["candidate_pool"]),
            keyword_weight=_env_float("RECALL_W_KEYWORD", d["keyword_weight"]),
            semantic_weight=_env_float("RECALL_W_SEMANTIC", d["semantic_weight"]),
            recency_weight=_env_float("RECALL_W_RECENCY", d["recency_weight"]),
            salience_weight=_env_float("RECALL_W_SALIENCE", d["salience_weight"]),
            recency_half_life_hours=_env_float("RECALL_HALF_LIFE_H", d["recency_half_life_hours"]),
            mmr_lambda=_env_float("RECALL_MMR_LAMBDA", d["mmr_lambda"]),
            include_archived=_env_bool("RECALL_INCLUDE_ARCHIVED", d["include_archived"]),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for the whole engine."""

    # --- model ---
    model: str = "ollama:llama3.1"
    """Provider-qualified model spec, e.g. ``ollama:llama3.1``,
    ``openai:gpt-4o-mini``, ``anthropic:claude-sonnet-4-5``, ``echo:test``."""

    temperature: float = 0.4
    max_output_tokens: int = 1024
    request_timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 1.5

    # --- provider endpoints / credentials ---
    ollama_host: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_api_key: str | None = None

    # --- persistence ---
    data_dir: Path = DEFAULT_DATA_DIR
    database_path: Path = DEFAULT_DATA_DIR / "crucible.sqlite3"

    # --- memory behaviour ---
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    context_token_budget: int = 6000
    """Soft cap on tokens of recalled memory injected into a single prompt."""

    compaction_threshold_tokens: int = 12000
    """Once a topic's live entries exceed this, compaction folds the oldest
    entries into a durable summary."""

    compaction_keep_recent: int = 8
    """Entries newer than this count are never compacted away."""

    embeddings_enabled: bool = True
    embedding_dims: int = 256

    # --- orchestration ---
    max_parallel_agents: int = 4
    debate_rounds: int = 3
    debate_convergence_threshold: float = 0.92
    """Cosine similarity between consecutive rounds above which a debate is
    considered converged and stops early."""

    tools_enabled: bool = True
    max_tool_calls_per_turn: int = 4
    web_search_enabled: bool = False

    # --- observability ---
    log_level: str = "INFO"
    log_json: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "database_path", Path(self.database_path))

    # -- derived -------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """The provider half of :attr:`model` (``"ollama"`` for ``ollama:llama3.1``)."""
        return split_model_spec(self.model)[0]

    @property
    def model_name(self) -> str:
        """The model half of :attr:`model`."""
        return split_model_spec(self.model)[1]

    # -- construction --------------------------------------------------------

    @classmethod
    def from_env(
        cls, *, dotenv: str | os.PathLike[str] | None = ".env", **overrides: Any
    ) -> Settings:
        """Build settings from ``.env`` + environment, then apply ``overrides``.

        Unknown override keys raise :class:`ConfigurationError` rather than being
        silently dropped, which is the failure mode that costs the most debugging
        time in config-heavy systems.
        """
        if dotenv is not None:
            load_dotenv(dotenv)

        d = _defaults(cls)
        data_dir = Path(_env("DATA_DIR", str(DEFAULT_DATA_DIR)) or DEFAULT_DATA_DIR)
        db_default = data_dir / "crucible.sqlite3"

        base = cls(
            model=_env("MODEL", d["model"]) or d["model"],
            temperature=_env_float("TEMPERATURE", d["temperature"]),
            max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", d["max_output_tokens"]),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", d["request_timeout_s"]),
            max_retries=_env_int("MAX_RETRIES", d["max_retries"]),
            retry_backoff_s=_env_float("RETRY_BACKOFF_S", d["retry_backoff_s"]),
            ollama_host=_env("OLLAMA_HOST", d["ollama_host"]) or d["ollama_host"],
            openai_base_url=_env("OPENAI_BASE_URL", d["openai_base_url"]) or d["openai_base_url"],
            openai_api_key=_env("OPENAI_API_KEY"),
            anthropic_base_url=(
                _env("ANTHROPIC_BASE_URL", d["anthropic_base_url"]) or d["anthropic_base_url"]
            ),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            data_dir=data_dir,
            database_path=Path(_env("DATABASE_PATH", str(db_default)) or db_default),
            retrieval=RetrievalSettings.from_env(),
            context_token_budget=_env_int("CONTEXT_TOKEN_BUDGET", d["context_token_budget"]),
            compaction_threshold_tokens=_env_int(
                "COMPACTION_THRESHOLD_TOKENS", d["compaction_threshold_tokens"]
            ),
            compaction_keep_recent=_env_int("COMPACTION_KEEP_RECENT", d["compaction_keep_recent"]),
            embeddings_enabled=_env_bool("EMBEDDINGS_ENABLED", d["embeddings_enabled"]),
            embedding_dims=_env_int("EMBEDDING_DIMS", d["embedding_dims"]),
            max_parallel_agents=_env_int("MAX_PARALLEL_AGENTS", d["max_parallel_agents"]),
            debate_rounds=_env_int("DEBATE_ROUNDS", d["debate_rounds"]),
            debate_convergence_threshold=_env_float(
                "DEBATE_CONVERGENCE", d["debate_convergence_threshold"]
            ),
            tools_enabled=_env_bool("TOOLS_ENABLED", d["tools_enabled"]),
            max_tool_calls_per_turn=_env_int("MAX_TOOL_CALLS", d["max_tool_calls_per_turn"]),
            web_search_enabled=_env_bool("WEB_SEARCH_ENABLED", d["web_search_enabled"]),
            log_level=(_env("LOG_LEVEL", d["log_level"]) or d["log_level"]).upper(),
            log_json=_env_bool("LOG_JSON", d["log_json"]),
        )
        settings = base.with_overrides(**overrides) if overrides else base
        settings.validate()
        return settings

    def with_overrides(self, **overrides: Any) -> Settings:
        known = {f.name for f in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ConfigurationError(
                f"unknown setting(s): {', '.join(unknown)}",
                hint=f"Valid settings: {', '.join(sorted(known))}",
            )
        return replace(self, **overrides)

    def validate(self) -> None:
        split_model_spec(self.model)  # raises on a malformed spec
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError("temperature must be within [0, 2]")
        if self.max_output_tokens < 1:
            raise ConfigurationError("max_output_tokens must be >= 1")
        if self.max_parallel_agents < 1:
            raise ConfigurationError("max_parallel_agents must be >= 1")
        if self.debate_rounds < 1:
            raise ConfigurationError("debate_rounds must be >= 1")
        if self.embedding_dims < 16:
            raise ConfigurationError("embedding_dims must be >= 16")
        if self.compaction_keep_recent < 1:
            raise ConfigurationError("compaction_keep_recent must be >= 1")
        self.retrieval.validate()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, Any]:
        """A dict safe to log or render in a UI: secrets become ``"set"``/``None``."""
        data = asdict(self)
        for key in ("openai_api_key", "anthropic_api_key"):
            data[key] = "set" if data.get(key) else None
        data["data_dir"] = str(self.data_dir)
        data["database_path"] = str(self.database_path)
        return data


def split_model_spec(spec: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into its parts.

    A spec with no provider prefix is assumed to be Ollama, which keeps the
    zero-config local path short::

        >>> split_model_spec("ollama:llama3.1")
        ('ollama', 'llama3.1')
        >>> split_model_spec("llama3.1")
        ('ollama', 'llama3.1')
        >>> split_model_spec("openai:gpt-4o-mini")
        ('openai', 'gpt-4o-mini')
    """
    if not spec or not spec.strip():
        raise ConfigurationError("model spec is empty", hint="Try CRUCIBLE_MODEL=ollama:llama3.1")
    spec = spec.strip()
    provider, sep, name = spec.partition(":")
    if not sep:
        return "ollama", spec
    provider, name = provider.strip().lower(), name.strip()
    if not name:
        raise ConfigurationError(
            f"model spec {spec!r} has a provider but no model name",
            hint="Use provider:model, e.g. ollama:llama3.1",
        )
    return provider, name
