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

`extract_pdfs.py` can extract with either `pdfplumber` or [LiteParse](https://github.com/run-llama/liteparse) — both are real dependencies and either works for most sources (`corpus.extract_pdf_pages` and `corpus.extract_pdf_pages_liteparse` produce the same output shape and work with the same segmentation functions). The default split (`EXTRACTOR` in `extract_pdfs.py`) uses LiteParse for `mlr_2017`/`jmlsg_1`/`jmlsg_2`/`fca_fcg` and `pdfplumber` for `fatf_40` specifically, since LiteParse's reading-order reconstruction breaks down on that one document's styled headings while `pdfplumber` handles it correctly — validated against the real corpus 2026-07-05, full comparison in `01_corpus_and_retrievers.md`'s 2026-07-05 Change log entry.

Run the notebooks in order, selecting the `AML RAG (Python 3.11)` kernel in each. Logs are written to `logs/` (not committed).

| # | Notebook | Phase | Covers |
|---|---|---|---|
| 01 | `01_corpus_and_retrievers.ipynb` | 1 | Load `clauses.jsonl`, build the ChromaDB and BM25 indexes |
| 02 | `02_generation_pipeline.ipynb` | 1 | LangGraph pipeline: dense+sparse retrieval, citation validation, guardrails |
| 03 | `03_evaluation_baseline.ipynb` | 1 | IR ablation (dense, sparse, dense+sparse); writes `results/baseline_ablation.csv` |
| 04 | `04_knowledge_graph.ipynb` | 2 | Cross-reference extraction, NetworkX graph, hop-count experiment |
| 05 | `05_hybrid_retrieval.ipynb` | 2 | Hybrid RRF retriever (dense + sparse + graph), smoke test |
| 06 | `06_evaluation_hybrid.ipynb` | 2 | Hybrid evaluation and comparison against the baseline |

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

## Dependencies

Managed with `uv`: `uv add <pkg>==<version>` for runtime, `uv add --dev <pkg>==<version>` for dev tools, `uv remove <pkg>` to remove one. All three update `pyproject.toml` and `uv.lock` together — never hand-edit those arrays directly, or the lockfile silently drifts out of sync.

To check (or force) the venv matches `pyproject.toml`/`uv.lock` exactly:

```bash
uv sync --all-groups
```

Unlike `uv add`/`pip install`, this also removes anything installed that isn't declared, so it catches drift in both directions. Running it twice in a row should be a no-op ("Checked N packages," nothing installed/uninstalled) — if it isn't, that's a sign of leftover debris in `.venv` worth investigating.
