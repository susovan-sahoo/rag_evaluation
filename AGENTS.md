# AGENTS.md

Guidance for AI coding agents working in this repo. See [README.md](README.md) for full setup/usage.

## Project overview

Two halves:

- **RAG pipeline** (`src/`): `retriever.py` builds/loads a Chroma vector store from `data/*.vtt` transcripts and exposes a plain k=5 retriever; `reranker.py` wraps it with a local CPU cross-encoder (`RerankingRetriever`) to over-fetch then rerank; `generator.py` answers a query from context chunks with a faithfulness-first prompt; `rag_pipeline.py`'s `RagPipeline` wires reranker + generator into one retrieve → generate call.
- **Evaluation harness** (`evals/` + `goldens/`): `eval_retriever.py`/`eval_retriever_with_reranker.py` score retrieval against `goldens/retriever_goldens.json`. `eval_generator.py` scores the generator in isolation — fed golden `ideal_context` directly from `goldens/faithfulness_dataset.json`, so a low score is purely the generator's fault. `eval_rag_pipeline.py` scores the full live pipeline against the same golden queries with triad metrics.

## Build / run

Same commands on every OS (macOS/Linux/Windows). Always use module form (`-m`) — see gotcha below:

```bash
uv sync
uv run python -m evals.eval_retriever
uv run python -m evals.eval_retriever_with_reranker
uv run python -m evals.eval_generator
uv run python -m evals.eval_rag_pipeline
```

## Conventions and gotchas

- `evals/*.py` are plain scripts, not pytest tests — they execute top-to-bottom on import (no `if __name__ == "__main__"`, no `test_` functions).
- **Always run evals with `-m`**, e.g. `uv run python -m evals.eval_retriever` — never `uv run python evals/eval_retriever.py`. This project has no `[build-system]` (unpackaged/"virtual" uv project), so `src` isn't installed into the venv and is only importable when the repo root is on `sys.path`, which `python -m` guarantees and direct script execution does not.
- `chroma_store/` and `.cache/` are generated and gitignored — never hand-edit or commit them. Delete `chroma_store/` to force a rebuild after changing chunking or the embedding model.
- `OPENAI_API_KEY` is required (embeddings, generation, DeepEval's LLM judge) and must come from `.env` (see `.env.example`). Never hardcode or commit keys.
- Any new entry in `goldens/retriever_goldens.json` must keep the exact keys `id`, `query`, `ideal_answer`, `source`. Any new entry in `goldens/faithfulness_dataset.json` must keep `id`, `query`, `ideal_context`, `source_sessions` — the eval scripts read these by name.
- The embedding model used in `src/retriever.py` (`text-embedding-3-small`) must match the `hyperparameters["embedding_model"]` value logged in the retriever eval scripts — keep them in sync if either changes.
- `src/generator.py` and both `eval_generator.py`/`eval_rag_pipeline.py` use `gpt-4o-mini` for generation and judging; the retriever evals use `gpt-4.1-mini` as judge only — this split is intentional, not a bug.
