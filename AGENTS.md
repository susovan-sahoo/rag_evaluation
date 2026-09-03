# AGENTS.md

Guidance for AI coding agents working in this repo. See [README.md](README.md) for full setup/usage.

## Project overview

Two halves:

- **RAG pipeline** (`src/`): `retriever.py` builds/loads a Chroma vector store from `data/*.vtt` transcripts and exposes a plain k=5 retriever; `reranker.py` wraps it with a local CPU cross-encoder (`RerankingRetriever`) to over-fetch then rerank; `generator.py` answers a query from context chunks with a faithfulness-first prompt; `rag_pipeline.py`'s `RagPipeline` wires reranker + generator into one retrieve → generate call.
- **Evaluation harness** (`evals/` + `goldens/`), run at three levels — component, pipeline, application:
  - Component: `eval_retriever.py`/`eval_retriever_with_reranker.py` score retrieval against `goldens/retriever_goldens.json`. `eval_generator.py` scores the generator in isolation — fed golden `ideal_context` directly from `goldens/faithfulness_dataset.json`, so a low score is purely the generator's fault.
  - Pipeline: `eval_rag_pipeline.py` scores the full live pipeline against the same golden queries with triad metrics.
  - Application: full live `RagPipeline` runs scored across the three risk categories — quality (`eval_application.py`: Correctness/Completeness/Style vs. `goldens/correctness_goldens.json`), safety (`eval_toxicity.py` vs. `goldens/toxicity_goldens.json`, `eval_leakage.py` vs. `goldens/leakage_goldens.json`, `eval_scope_safety.py` vs. `goldens/scope_goldens.json`), and operations (`eval_latency.py`, `eval_cost.py`, `eval_reliability.py` — no golden dataset or LLM judge, scored against hardcoded SLO/budget constants instead).

## Build / run

Same commands on every OS (macOS/Linux/Windows). Always use module form (`-m`) — see gotcha below:

```bash
uv sync
uv run python -m evals.eval_retriever
uv run python -m evals.eval_retriever_with_reranker
uv run python -m evals.eval_generator
uv run python -m evals.eval_rag_pipeline
uv run python -m evals.eval_application
uv run python -m evals.eval_toxicity
uv run python -m evals.eval_leakage
uv run python -m evals.eval_scope_safety
uv run python -m evals.eval_latency
uv run python -m evals.eval_cost
uv run python -m evals.eval_reliability
```

## Conventions and gotchas

- `evals/*.py` are plain scripts, not pytest tests — they execute top-to-bottom on import (no `if __name__ == "__main__"`, no `test_` functions).
- **Always run evals with `-m`**, e.g. `uv run python -m evals.eval_retriever` — never `uv run python evals/eval_retriever.py`. This project has no `[build-system]` (unpackaged/"virtual" uv project), so `src` isn't installed into the venv and is only importable when the repo root is on `sys.path`, which `python -m` guarantees and direct script execution does not.
- `chroma_store/` and `.cache/` are generated and gitignored — never hand-edit or commit them. Delete `chroma_store/` to force a rebuild after changing chunking or the embedding model.
- `OPENAI_API_KEY` is required (embeddings, generation, DeepEval's LLM judge) and must come from `.env` (see `.env.example`). Never hardcode or commit keys.
- Any new entry in `goldens/retriever_goldens.json` must keep the exact keys `id`, `query`, `ideal_answer`, `source`. Any new entry in `goldens/faithfulness_dataset.json` must keep `id`, `query`, `ideal_context`, `source_sessions` — the eval scripts read these by name.
- The embedding model used in `src/retriever.py` (`text-embedding-3-small`) must match the `hyperparameters["embedding_model"]` value logged in the retriever eval scripts — keep them in sync if either changes.
- `src/generator.py` and both `eval_generator.py`/`eval_rag_pipeline.py` use `gpt-4o-mini` for generation and judging; the retriever evals use `gpt-4.1-mini` as judge only — this split is intentional, not a bug.
- Each new application-level goldens file has its OWN key schema — read keys per file, don't assume `retriever_goldens.json`'s `query`/`source` shape: `goldens/correctness_goldens.json` uses `id`, `question`, `ideal_answer`, `source_session`; `goldens/toxicity_goldens.json` uses `id`, `case_type`, `technique`, `input`; `goldens/leakage_goldens.json` uses `id`, `subtype`, `case_type`, `technique`, `input`, plus `expected_action` (present for `subtype: prompt`/`course_content`, absent for `subtype: pii` since `PIILeakageMetric` needs no expected output); `goldens/scope_goldens.json` uses `id`, `case_type`, `technique`, `input`, `expected_action`, `success_criteria`.
- Threshold direction varies by metric — don't assume "0.7, higher passes" repo-wide: `GEval`-based metrics (Correctness/Completeness/Style/Scope Adherence/Prompt & Course Content Leakage) score 0–10 with threshold=0.7, higher-is-better; `ToxicityMetric` (`eval_toxicity.py`) uses threshold=0.3, LOWER-is-better (passes when toxicity <= 0.3); `PIILeakageMetric` (`eval_leakage.py`) uses threshold=0.9, higher-is-better (passes when leakage-avoidance score >= 0.9).
- `eval_latency.py`/`eval_cost.py`/`eval_reliability.py` are the operational trio — no `goldens/*.json`, no LLM judge; they run a hardcoded `QUESTIONS` list and score against hardcoded SLO/budget constants in each file (`SLO_P95_MS`, `SLO_TTFT_P95_MS`, `PRICE_*_PER_1M`, `QUERIES_PER_DAY`, `USD_TO_INR`, `COST_BUDGET_PER_QUERY_USD`, `REPEATS`, `MAX_RETRIES`, `BACKOFF_BASE_S`) — tune those constants directly, there's no external config file.
- `eval_cost.py` imports `prompt`/`llm` directly from `src/generator.py` (bypassing `generate()`'s `StrOutputParser`) to read token `usage_metadata` off the raw `AIMessage` — renaming/removing those module-level names in `generator.py` breaks `eval_cost.py` at import time.
- `src/generator.py`'s prompt is parsed as an f-string by `ChatPromptTemplate.from_template` — any edit to the safety-rules text must not introduce a stray `{`/`}` outside the `{context}`/`{question}` placeholders, or the template fails at construction time, not at call time.
- New `GEval` metrics follow `from deepeval.metrics.g_eval import Rubric` + `GEval(rubric=[Rubric(score_range=(a, b), expected_outcome=...), ...])`; `score_range` values must be integers within 0–10 with start<=end (pydantic-validated). Note `LLMTestCaseParams` (used for `evaluation_params`) is deprecated in favor of `SingleTurnParams` in deepeval 4.1.5 — still works, but may be removed in a future release.
