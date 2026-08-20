# RAG evaluation using DeepEval

Evaluation harness for a transcript-based RAG pipeline (retrieval, reranking, generation) using [DeepEval](https://github.com/confident-ai/deepeval) metrics.

The retrieval corpus is a set of `.vtt` transcripts (in `data/`) from an LLM evals course. The project builds a Chroma vector store from those transcripts and evaluates it at three levels: retrieval quality (plain vs. reranked), generator faithfulness in isolation, and the full retrieve → rerank → generate pipeline — each against a hand-authored golden dataset.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key (used for embeddings and as the DeepEval LLM judge)

## Setup

Install dependencies into a local venv (uses the committed `uv.lock`), then create your local env file:

**macOS / Linux (bash/zsh):**

```bash
uv sync
cp .env.example .env
```

**Windows (PowerShell):**

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY=sk-...`.

## Project structure

```text
data/                               8 .vtt transcripts (the corpus)
src/retriever.py                    loads transcripts, builds/loads the Chroma store, plain k=5 retriever
src/reranker.py                     cross-encoder reranker on top of the base retriever (local, no API key)
src/generator.py                    generates a grounded answer from a query + context chunks
src/rag_pipeline.py                 RagPipeline: retrieve (reranked) -> generate, in one call
evals/eval_retriever.py             DeepEval run: base retriever vs. golden dataset
evals/eval_retriever_with_reranker.py  DeepEval run: reranked retriever vs. golden dataset
evals/eval_generator.py             DeepEval run: generator in isolation vs. golden context
evals/eval_rag_pipeline.py          DeepEval run: full pipeline vs. golden queries
goldens/retriever_goldens.json      hand-authored queries + ideal answers for the retriever evals
goldens/faithfulness_dataset.json   hand-authored queries + ideal context for the generator/pipeline evals
main.py                             placeholder entry point (not part of the eval flow)
```

## Running the evaluations

Run every script from the repo root using module form (`-m`) — this project has no `[build-system]`, so `src` is only importable when the repo root is on `sys.path`, which `-m` guarantees:

```bash
uv run python -m evals.eval_retriever
uv run python -m evals.eval_retriever_with_reranker
uv run python -m evals.eval_generator
uv run python -m evals.eval_rag_pipeline
```

- `eval_retriever` / `eval_retriever_with_reranker`: score retrieval against `goldens/retriever_goldens.json` with `ContextualRecallMetric` + `ContextualPrecisionMetric` (judge: `gpt-4.1-mini`).
- `eval_generator`: feeds the generator the golden `ideal_context` directly (isolated from retrieval) and scores it with `FaithfulnessMetric` + `AnswerRelevancyMetric` (judge: `gpt-4o-mini`).
- `eval_rag_pipeline`: runs the full retrieve → rerank → generate pipeline per query and scores it with `ContextualRelevancyMetric` + `FaithfulnessMetric` + `AnswerRelevancyMetric` (judge: `gpt-4o-mini`).

All scripts use threshold 0.7 and print a pass/fail report to the console.

Notes:

- The first run builds `chroma_store/` from `data/*.vtt` (chunked, embedded with `text-embedding-3-small`) and persists it to disk — later runs reuse it. Delete `chroma_store/` to force a rebuild (e.g. after changing chunking or the embedding model).
- The reranker-based scripts also download a small local cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) on first use — no API key needed for that part.
- Every script makes OpenAI calls (embeddings and/or generation, plus the LLM judge); `eval_rag_pipeline` is the most expensive (embeddings + rerank + generation + 3 judge calls per query).

## Golden dataset

`goldens/retriever_goldens.json` is a list of entries shaped like:

```json
{
  "id": "g001",
  "query": "What is online eval and how is it different from offline eval?",
  "ideal_answer": "...",
  "source": "Session 4"
}
```

To add a case, append an entry with those same four keys — the eval scripts read `query` and `ideal_answer` directly.

`goldens/faithfulness_dataset.json` (used by `eval_generator` and `eval_rag_pipeline`) is shaped like:

```json
{
  "id": "f001",
  "query": "what is vibe testing and why isn't it enough for deploying an llm app?",
  "ideal_context": ["chunk 1", "chunk 2"],
  "source_sessions": ["1"]
}
```

`eval_generator` reads `query` + `ideal_context`; `eval_rag_pipeline` reads only `query` (context comes live from the pipeline).

## Working with AI coding agents

See [AGENTS.md](AGENTS.md) for architecture notes and conventions aimed at AI coding assistants working in this repo.
