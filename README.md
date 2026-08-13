# RAG evaluation using DeepEval

Evaluation harness for a transcript-based RAG retriever using [DeepEval](https://github.com/confident-ai/deepeval) metrics.

The retrieval corpus is a set of `.vtt` transcripts (in `data/`) from an LLM evals course. The project builds a Chroma vector store from those transcripts and scores two retrieval strategies — a plain vector-similarity retriever and a reranked retriever — against a hand-authored golden dataset.

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
evals/eval_retriever.py             DeepEval run: base retriever vs. golden dataset
evals/eval_retriever_with_reranker.py  DeepEval run: reranked retriever vs. golden dataset
goldens/retriever_goldens.json      hand-authored queries + ideal answers used as the golden set
main.py                             placeholder entry point (not part of the eval flow)
```

## Running the evaluations

Same commands on every OS:

```bash
uv run python evals/eval_retriever.py
uv run python evals/eval_retriever_with_reranker.py
```

Each script loads `goldens/retriever_goldens.json`, retrieves context for every query, and scores the result with DeepEval's `ContextualRecallMetric` and `ContextualPrecisionMetric` (LLM judge: `gpt-4.1-mini`, threshold 0.7). A pass/fail report prints to the console.

Notes:

- The first run builds `chroma_store/` from `data/*.vtt` (chunked, embedded with `text-embedding-3-small`) and persists it to disk — later runs reuse it. Delete `chroma_store/` to force a rebuild (e.g. after changing chunking or the embedding model).
- The reranker script also downloads a small local cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) on first use — no API key needed for that part.
- Both scripts make OpenAI calls (embeddings + LLM judge), so running them repeatedly has a small cost.

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

## Working with AI coding agents

See [AGENTS.md](AGENTS.md) for architecture notes and conventions aimed at AI coding assistants working in this repo.
