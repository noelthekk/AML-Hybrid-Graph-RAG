# AML Hybrid RAG: Implementation

Hybrid retrieval-augmented generation for UK AML regulatory compliance. Six source documents are segmented into 2,568 clauses, retrieved with dense + sparse + graph fusion (RRF), and answered by Llama 3.1 via LangGraph.

## Setup

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
cd plan/implementation
uv sync --all-groups
.venv\Scripts\python -m ipykernel install --user --name aml-rag --display-name "AML RAG (Python 3.11)"
.venv\Scripts\python verify_env.py
```

Copy `.env.example` to `.env` and set `HF_TOKEN` (needed only for notebook 04 and RAGAS evaluation).

## Running order

Place the corpus PDFs in `data/raw/` (see below), then cache the text and launch Jupyter:

```bash
.venv\Scripts\python scripts\extract_pdfs.py   # cache raw text to data/interim/
.venv\Scripts\python start_jupyter.py          # serves at http://localhost:8888
```

`extract_pdfs.py` extracts with either `pdfplumber` or [LiteParse](https://github.com/run-llama/liteparse) per source (both real dependencies, same output shape) — see its `EXTRACTOR` mapping for the current split, and `01_corpus_and_retrievers.md`'s 2026-07-05 Change log entry for why `fatf_40` stays on `pdfplumber` specifically.

Run the notebooks in order, selecting the `AML RAG (Python 3.11)` kernel in each. Logs are written to `logs/` (not committed).

| # | Notebook | Phase | Covers |
|---|---|---|---|
| 01 | `01_corpus_and_retrievers.ipynb` | 1 | Load `clauses.jsonl`, build the ChromaDB and BM25 indexes |
| 02 | `02_generation_pipeline.ipynb` | 1 | LangGraph pipeline: dense+sparse retrieval, citation validation, guardrails |
| 03 | `03_evaluation_baseline.ipynb` | 1 | IR ablation (dense, sparse, dense+sparse); writes `results/baseline_ablation.csv` |
| 04 | `04_knowledge_graph.ipynb` | 2 | Cross-reference extraction, NetworkX graph, hop-count experiment |
| 05 | `05_hybrid_retrieval.ipynb` | 2 | Hybrid RRF retriever (dense + sparse + graph), smoke test |
| 06 | `06_evaluation_hybrid.ipynb` | 2 | Hybrid evaluation and comparison against the baseline |
| 07 | `07_ragas_evaluation.ipynb` | 2 | RAGAS + correctness evaluation across all four configs (in progress) |

## Corpus PDFs

Save all files to `data/raw/`. Three download automatically when running notebook 01; three need manual download.

| Document | File | Source |
|---|---|---|
| MLR 2017 | `mlr_2017.pdf` | Auto-downloaded |
| POCA 2002 | `poca_2002.pdf` | Auto-downloaded (XML also fetched from legislation.gov.uk) |
| FATF 40 | `fatf_40.pdf` | Auto-downloaded |
| JMLSG Part I | `jmlsg_part1.pdf` | [jmlsg.org.uk](https://www.jmlsg.org.uk/guidance/), free, requires name/email |
| JMLSG Part II | `jmlsg_part2.pdf` | [jmlsg.org.uk](https://www.jmlsg.org.uk/guidance/), free, requires name/email |
| FCA FCG | `fca_fcg.pdf` | [handbook.fca.org.uk](https://www.handbook.fca.org.uk/handbook/FCG.pdf) |

## Experiments

Two folders hold generation-side experiments run against the 21 "universal failure"
queries (queries that scored 0 correctness across every retrieval config, both
reference models, and both generator scales in the primary 200-query evaluation):

- `experiment/` — failure-mode analysis, a judge-reliability diagnostic, and two
  prompting/pipeline fixes on the local Llama-3.1-8B generator:
  `p1_failure_analysis.ipynb` (finds the dominant failure mode is truncation of
  multi-part clause content), `correctness_reasoning_diagnostic.py` (checks whether the
  `llm_grade` correctness judge's "no" verdicts are arbitrary or genuinely defensible —
  finds every one it sampled was a real, specific content gap, not judge unreliability),
  `p2_structured_prompting.ipynb` (completeness-instruction prompt fix), and
  `p2_two_step_extraction.ipynb` (extraction-then-synthesis pipeline fix). `p1`/`p2`
  each have a matching `.py` script the notebook was built from. See its own `README.md`
  for conventions. Nothing here is validated pipeline output — treat as scratch until an
  idea is proven out and folded back into notebook 02/07 proper.
- `gptTest/` — swaps the generator to `gpt-5-nano` (OpenAI API, with and without
  reasoning effort) across all four retrieval configs, compared against the `experiment/`
  results above. See `gptTest/plan.md` for the design and `info.md` for the full
  setup-by-setup comparison table.

Data used by both: `results/p1_universal_failures.jsonl` (the 21 queries, with each
config's already-retrieved clause IDs — retrieval is never re-run by these experiments),
`data/clauses.jsonl` (clause text lookup), and `data/test_set.jsonl` (`reference_14b`/
`reference_72b` answers to score against). `gptTest/` additionally reads the existing
baseline/P2 scores for its comparison table: `results/correctness_scores.jsonl`,
`results/ragas_answer_scores.jsonl`, `results/ragas_retrieval_scores.jsonl`,
`results/p2_structured_prompting_scores.jsonl`, `results/p2v2_two_step_scores.jsonl`.
Each experiment's own generated answers/scores are written locally to its own folder
(`experiment/`'s outputs go to the shared top-level `results/`; `gptTest/`'s outputs stay
self-contained in `gptTest/results/`).

## Dependencies

Managed with `uv`: `uv add <pkg>==<version>` for runtime, `uv add --dev <pkg>==<version>` for dev tools, `uv remove <pkg>` to remove one. Never hand-edit `pyproject.toml`'s dependency arrays directly, or `uv.lock` silently drifts out of sync.

To check (or force) the venv matches `pyproject.toml`/`uv.lock` exactly:

```bash
uv sync --all-groups
```

Unlike `uv add`/`pip install`, this also removes anything installed that isn't declared, catching drift in both directions. Running it twice in a row should be a no-op — if it isn't, that's leftover `.venv` debris worth investigating.
