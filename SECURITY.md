# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability.

Use GitHub's [private vulnerability reporting](https://github.com/Abdur-Rafay-AR/athenacore-multiagent/security/advisories/new),
or contact the maintainer through their GitHub profile. Include what the issue is,
how to reproduce it, and what an attacker could achieve. Expect an initial response
within a few days.

## Supported versions

The latest release on `main` receives fixes. This is a young project; there are no
long-term support branches.

## Threat model

AthenaCore is a **local-first, single-user tool**. It assumes the operator trusts
the machine it runs on and the model it is pointed at. That assumption shapes what
counts as a vulnerability here.

### Where hardening exists

- **Expression evaluation.** The calculator tool walks a whitelisted AST and never
  calls `eval`. Attribute access, imports, comprehensions and arbitrary calls are
  rejected; exponents are capped to prevent resource exhaustion.
- **SQL.** Every query uses parameter binding. Free-text search is sanitised into a
  quoted FTS5 `MATCH` expression rather than interpolated, which prevents both
  injection and the crashes malformed match syntax would otherwise cause.
- **Side-effecting tools are opt-in.** Tools with cost, side effects or network
  egress set `safe = False` and stay disabled unless explicitly enabled. Web search
  is off by default.
- **Secrets.** API keys are read from the environment, never persisted to the
  database, and redacted by `Settings.redacted()` and the `/config` endpoint.
- **HTML rendering.** All model-generated content is escaped before being rendered
  in the UI.

### Known limitations, by design

- **The API has no authentication.** `athenacore serve` binds to `127.0.0.1` and
  sends permissive CORS headers, because it is meant for local use. **Do not expose
  it to a network** without putting authentication and a tightened CORS policy in
  front of it.
- **Prompt injection is not solved.** Content in memory, including anything the web
  search tool retrieves, is fed to models as context. Untrusted retrieved text can
  influence agent behaviour. Treat memory content as untrusted input, and leave web
  search off unless you need it.
- **Model output is not sandboxed beyond the tool layer.** Tools constrain what a
  model can *do*; nothing constrains what it can *say*. Do not add a tool with
  destructive capability and assume prompt instructions will restrain it.
- **The database is unencrypted.** It is a plain SQLite file with filesystem
  permissions as its only protection. Do not store secrets in memory entries.

## Hardening for shared deployment

If you must run this beyond your own machine:

1. Put the API behind a reverse proxy with authentication.
2. Replace the wildcard CORS policy in `api/server.py`.
3. Set `ATHENA_WEB_SEARCH_ENABLED=false` and leave write-capable tools disabled.
4. Run as an unprivileged user with the database on a restricted volume.
5. Rate-limit run creation; a single request can trigger many model calls.
