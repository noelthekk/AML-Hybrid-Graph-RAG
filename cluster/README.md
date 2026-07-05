# Reference-Answer Drafting (Cluster)

Self-contained project to draft reference answers for `data/test_set.jsonl`, using
Qwen2.5-Instruct (NF4 4-bit). Supports either model size via `--model-size`, since which
one you can run depends on which VRAM partition you can get — see Run below. Not part of
the main `aml-hybrid-rag` project — has its own dependencies, meant to be copied to and
run on the cluster independently. `data/test_set.jsonl` and `data/clauses.jsonl` are
already included (copied from the main project 2026-07-02) — nothing to fetch before
transferring this folder.

## Setup

1. Copy this whole `cluster/` folder to the cluster:
   ```bash
   scp -r "C:/Users/tsono/Documents/uoe/disertation/plan/implementation/cluster" username@remote_host:/path/to/remote/directory/
   ```
2. Copy `.env.example` to `.env` and set `HF_TOKEN` (needed to download the model).
3. Install `uv` if not already present: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. `uv sync`

If the default `torch` install doesn't pick up GPU support for your cluster's CUDA
version, check `nvidia-smi` and install the matching `torch` build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before
`uv sync`.

## Run

```bash
uv run python generate_references.py --model-size 14b   # needs an 18GB partition
uv run python generate_references.py --model-size 72b   # needs a 72GB partition
```

Progress logs to the terminal with timestamps as it goes.

`--model-size` is required — there's no default, so you can't accidentally run the
wrong size against whatever partition you actually requested. 72b was always the
originally-preferred size (better quality — see `plan.md` Sec 5, "Reference-answer
drafting model" row); 14b is the fallback for when only an 18GB partition is available.
VRAM math for both:

| Size | Weights (NF4 4-bit) | Partition needed |
|---|---|---|
| 14b | ~7-8GB | 18GB (comfortable headroom) |
| 72b | ~36-40GB | 72GB (~30GB+ headroom) |

## Output

`results/reference_drafts_<model-size>.jsonl` (e.g. `reference_drafts_72b.jsonl`) — one
row per test-set query: `query`, `gold_ids`, `query_type` (from the input) plus
`draft_reference` and `citations` (drafted). The model-size suffix means a 14b run and a
72b run never overwrite each other, so you can compare both if you want. Rows where the
model's output wasn't valid JSON get a best-effort fallback and an `_parse_error` field
— check for those first.

An automated pre-check runs at the end and flags any row where `citations` doesn't
match `gold_ids` exactly (the two rows with 3-4 gold clauses are the likeliest spot for
this, regardless of model size).

Copy whichever `reference_drafts_<model-size>.jsonl` you actually want back to the main
project's `plan/implementation/results/` (rename to `reference_drafts.jsonl` there — the
local Phase 4/5/6 steps expect that name, not the size-suffixed one). The automated
pre-check has already run here; human verification (all 50 rows, no sampling) is the
next step — see `07_ragas_evaluation.md`'s completion criteria for the full remaining
checklist.
