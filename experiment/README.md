# Experiment Sandbox

Working folder for testing generation-side improvement ideas (decomposed/structured
prompting, few-shot exemplars, failure-mode analysis, and similar) one at a time.
Recreated 2026-07-09 (an earlier version existed for retrieval-ranking work, which
moved to `plan/generation_cluster/` instead — an independent repo outside this one,
not a subfolder here — see `dynamic_rrf/`/`query_type_retrieval/` there).

This folder reuses the main implementation's `data/`/`scripts/`/`results/` via relative
paths one level up (JupyterLab kernels run with cwd set to the *notebook's own folder*,
not the server root, so paths here need an explicit `../` where the main implementation
notebooks just say `data/`/`scripts/`). Nothing in here is validated pipeline output —
treat everything as scratch until an idea is proven out and (if it's a retrieval change)
moved to `plan/generation_cluster/`, or (if it's a generation/prompting change) folded
back into notebook 02/07 proper.

Each experiment gets its own notebook or subfolder here, numbered by priority (e.g.
`p1_failure_analysis.ipynb`, `p2_decomposed_prompting.ipynb`). Cross-encoder reranking
lives in `plan/generation_cluster/` instead, alongside `dynamic_rrf/`/`query_type_retrieval/`
— it's a retrieval-side change, not a generation/prompting one.
