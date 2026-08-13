# AGENTS.md

Guidance for AI coding agents working in this repo. See [README.md](README.md) for full setup/usage.

## Project overview

Two halves:

- **Retrieval pipeline** (`src/`): `retriever.py` builds/loads a Chroma vector store from `data/*.vtt` transcripts and exposes a plain k=5 retriever; `reranker.py` wraps it with a local CPU cross-encoder (`RerankingRetriever`) to over-fetch then rerank.
- **Evaluation harness** (`evals/` + `goldens/`): `evals/eval_retriever.py` and `evals/eval_retriever_with_reranker.py` run the golden queries in `goldens/retriever_goldens.json` through the respective retriever and score them with DeepEval metrics.

## Build / run

Same commands on every OS (macOS/Linux/Windows):

```bash
uv sync
uv run python evals/eval_retriever.py
uv run python evals/eval_retriever_with_reranker.py
```

## Conventions and gotchas

- `evals/*.py` are plain scripts, not pytest tests — they execute top-to-bottom on import (no `if __name__ == "__main__"`, no `test_` functions). Run them directly with `python`/`uv run python`.
- `chroma_store/` and `.cache/` are generated and gitignored — never hand-edit or commit them. Delete `chroma_store/` to force a rebuild after changing chunking or the embedding model.
- `OPENAI_API_KEY` is required (embeddings + DeepEval's LLM judge) and must come from `.env` (see `.env.example`). Never hardcode or commit keys.
- Any new entry added to `goldens/retriever_goldens.json` must keep the exact keys `id`, `query`, `ideal_answer`, `source` — the eval scripts read `query` and `ideal_answer` by name.
- The embedding model used in `src/retriever.py` (`text-embedding-3-small`) must match the `hyperparameters["embedding_model"]` value logged in both eval scripts — keep them in sync if either changes.
