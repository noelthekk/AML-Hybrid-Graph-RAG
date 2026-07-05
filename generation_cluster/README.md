# Answer Generation (Cluster)

Self-contained project to generate answers for all 50 test-set queries across all four
retrieval configs (`dense_only`, `sparse_only`, `dense_sparse`, `hybrid`) — 200 rows total
— using Llama-3.1-8B-Instruct in BF16. Not part of the main `aml-hybrid-rag` project — has
its own dependencies and its own copies of `scripts/retriever.py`, `scripts/graph_build.py`,
`scripts/build_chroma.py`, and the data files they need (`clauses.jsonl`,
`cross_refs.jsonl`, `test_set.jsonl`) — all copied from the main project 2026-07-05.
`data/chroma_db/` is *not* included (gitignored here, same as in the main project) — it's
a rebuildable binary index, not source data, so it's regenerated locally instead of
carried around as a ~38MB copy. Nothing else to fetch before transferring this folder.

## Why BF16, and why always the 8B model

This mirrors notebook 07's own generator setup exactly, and deliberately does **not**
offer a bigger-model option the way the reference-drafting `cluster/` folder does for
Qwen. Notebook 07 compares four retrieval configs against the same generated answers'
quality — if the generator changed between a local run (8B, NF4 4-bit) and a cluster run
(would-be 70B, BF16), retrieval-quality differences and generator-quality differences
would be entangled, which defeats the point of the comparison. Only *precision* differs
by hardware (NF4 4-bit locally where VRAM is tight, BF16 here where it isn't) — model
identity does not. Full reasoning: `plan.md` Sec 5, "Generator model identity" (this was
a real bug in an earlier version of notebook 02, caught and fixed 2026-07-05).

## Setup

1. Copy this whole `generation_cluster/` folder to the cluster:
   ```bash
   scp -r "C:/Users/tsono/Documents/uoe/disertation/plan/implementation/generation_cluster" username@remote_host:/path/to/remote/directory/
   ```
2. Copy `.env.example` to `.env` and set `HF_TOKEN` (needed to download the gated Llama
   weights).
3. Install `uv` if not already present: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. `uv sync`
5. Build the local vector index (not included in git, see above):
   ```bash
   uv run python scripts/build_chroma.py
   ```
   Takes a couple of minutes (CPU embeddings, `all-MiniLM-L6-v2`, 2,568 clauses). Safe to
   re-run — it loads and verifies the existing index instead of rebuilding if
   `data/chroma_db/` already has content.

If the default `torch` install doesn't pick up GPU support for your cluster's CUDA
version, check `nvidia-smi` and install the matching `torch` build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before
`uv sync`.

## Run

```bash
uv run python generate_answers.py
```

Progress logs to the terminal with timestamps, including a running per-query average and
an estimated time remaining. Based on a single-query timing test on a local 6GB GPU
(129.6s/query, NF4 4-bit) and a same-hardware-class reference point from the
reference-drafting run (Qwen2.5-72B, NF4 4-bit: 14.2s/query on this cluster), a rough
estimate for all 200 queries here is **20-40 minutes** — unverified, since BF16 on an 8B
model hasn't been timed directly on this hardware yet. The first few log lines will give
an actual per-query average early on.

## Output

`results/answers.jsonl` — 200 rows (4 configs x 50 queries): `query`, `gold_ids`,
`query_type`, `config`, `retrieved` (the clause IDs actually retrieved for that
query+config), `answer`, `citations`. Written incrementally (one line per row, flushed
immediately), so a crash partway through doesn't lose completed rows.

Rows where the model's output wasn't valid JSON get `answer` starting with
`"Generation error: ..."` and empty `citations` — check for those first; matches the
same fallback notebook 02's `generate_node` uses.

Copy `results/answers.jsonl` back to the main project's `plan/implementation/results/`
when done — that's the exact filename and location notebook 07's later cell groups
(retrieval-side and answer-side RAGAS) expect.
