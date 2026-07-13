# gptTest: GPT-5-nano as generator, thinking vs non-thinking

Plan for this notebook — read before opening `gpt5_generation_test.ipynb`, per the
project's MD-is-the-plan convention (`plan/CLAUDE.md`).

## Objective

Answer two linked questions on the 21 "universal failure" queries (queries that scored
`llm_grade=0` across all four retrieval configs, both reference models, and both
generator scales — see `plan/improvement_plan.md` P1, `results/p1_universal_failures.jsonl`):

1. Does swapping the generator from local `Llama-3.1-8B-Instruct` to an OpenAI
   `gpt-5-nano` change correctness/faithfulness/relevancy on these specifically-hard
   queries?
2. Does enabling extended reasoning ("thinking", via the `reasoning_effort` API
   parameter) help further — motivated by P1's finding that the dominant failure mode
   (~57%) is truncation of multi-part clause content, and by the separate finding that
   the cross-reference correctness gap is a synthesis problem, not a retrieval one. Both
   are exactly what explicit multi-step reasoning targets (enumerate every condition,
   hold multiple clauses in mind before writing the final answer).

This is a continuation of the P1/P2 experiment lineage (`experiment/p1_failure_analysis.py`,
`p2_structured_prompting.py`, `p2_two_step_extraction.py`), not a replacement for it —
same 21 queries, same retrieval, same scoring methodology, so results slot directly into
the existing comparison.

## What stays constant (for comparability)

Everything except the generator is held identical to the existing P2 variants, so scores
are directly comparable to numbers already in `results/correctness_scores.jsonl`,
`results/p2_structured_prompting_scores.jsonl`, and `results/p2v2_two_step_scores.jsonl`:

- **Retrieval**: not re-run. Reuses the `hybrid` config's `retrieved` clause IDs already
  recorded per query in `results/p1_universal_failures.jsonl` (`answers_8b.hybrid.retrieved`).
- **Reference answers**: `reference_14b` / `reference_72b` from `data/test_set.jsonl`
  (Qwen-drafted, unchanged). Not re-drafted with a GPT model — a fresh GPT-drafted
  reference would break comparability with every prior baseline/P2 number.
- **Judge**: `gpt-5.5` via the OpenAI API, exactly as `p2_structured_prompting.py` uses it
  (RAGAS `Faithfulness`/`AnswerRelevancy` + a custom cosine-similarity + yes/no
  `llm_grade` correctness check). Not swapped for `gpt-5`, even though the earlier
  "3 models, one family" discussion considered `gpt-5` as judge for a from-scratch
  same-family pipeline — that framing doesn't apply here, since this experiment's whole
  point is a controlled comparison against already-recorded numbers.
- **Context construction**: same clause-text truncation as `p2_structured_prompting.py`
  (`clause['text'][:800]` per retrieved clause, joined with `---`).

**What changes**: only the generator (`Llama-3.1-8B-Instruct` → `gpt-5-nano`) and the
reasoning effort applied to it. `gpt-5-nano` was chosen (not `gpt-5`/`gpt-5-mini`) to keep
the generator at a comparably modest capability tier to the 8B baseline — the earlier
same-family role assignment (`gpt-5`=judge, `gpt-5-mini`=reference, `gpt-5-nano`=generation)
reasoned that the system-under-test should stay modest, otherwise it trivially saturates
the correctness ceiling and tells us less about the failure modes we're actually probing.

## Conditions

Two generation conditions, both against the same 21 queries and same retrieved context:

| Condition | `reasoning_effort` | Label |
|---|---|---|
| Non-thinking | `"minimal"` (or omitted if unsupported — confirmed in the probe step) | `gpt5nano_base` |
| Thinking | `"high"` | `gpt5nano_thinking` |

One run per condition (not repeated) — this is an exploratory 21-query gate, matching the
scale of `dynamic_rrf`'s and `query_type_retrieval`'s cheap-gate pattern, not a full
200-row commitment. If either condition looks promising, scaling to the full 200-row test
set is a separate, later decision.

## Known risks / unknowns — resolved empirically, not assumed

- **Whether `gpt-5-nano` actually accepts `reasoning_effort`, and what values are valid.**
  Unconfirmed — this codebase has only ever called `gpt-5.5` as a fixed-effort judge, never
  set reasoning effort explicitly. First notebook cell group is a 1-2 row smoke test
  against the real API before running all 42 generation calls, mirroring how the RAGAS
  judge wrapper API was confirmed against the installed package before trusting it
  (`07_ragas_evaluation.md`).
- **Reasoning tokens silently consuming `max_completion_tokens`.** Already bit this
  project twice (the `gpt-5.5` judge and the correctness grader both returned empty
  replies at `max_completion_tokens=10`, needing `4096`, since reasoning tokens are
  deducted from the same budget before any visible output). Start at `4096`; the probe
  step checks for empty/truncated output before scaling up further if needed.
  Non-reasoning/minimal-effort condition may need a smaller budget — checked, not assumed.
- **JSON parsing.** Rather than relying on prompt instructions alone (as the local HF
  pipeline did, which needed a 145/200 recovery pass for the 70B ablation), use OpenAI's
  `response_format={"type": "json_object"}` — should eliminate the JSON-parsing failure
  mode for this generator entirely. Confirmed in the probe step, not assumed.
- **Non-determinism.** Unlike the local Llama runs (`do_sample=False`, fully
  deterministic — relied on elsewhere to treat generation errors as reproducible),
  OpenAI reasoning-tier models are not guaranteed deterministic even with a pinned
  `temperature`. One run per condition is treated as a single data point, not a stable
  ground truth — worth flagging in the write-up, not a blocker for this exploratory scale.
- **Cost/latency.** `gpt-5-nano` is the cheapest tier regardless of condition; 21 queries
  x 2 conditions is small. Thinking-mode calls will be slower and use more tokens
  (reasoning + output) — the probe step gives a per-call time/token estimate before
  running all 21.

## Folder layout (self-contained, per this session's instruction to log everything here)

```
gptTest/
  plan.md                        (this file)
  gpt5_generation_test.ipynb      (the notebook, built cell-group by cell-group)
  results/
    answers_nonthinking.jsonl     (21 rows, gpt5nano_base)
    answers_thinking.jsonl        (21 rows, gpt5nano_thinking)
    scores.jsonl                  (both conditions x both reference models)
    comparison.csv                (merged: baseline + p2 + p2v2 + both new conditions)
  figures/
    comparison.png
```

Unlike `experiment/`'s convention (writes into the shared top-level `results/`), this
folder's own outputs stay local to `gptTest/` — only the *inputs* (`p1_universal_failures.jsonl`,
`test_set.jsonl`, `clauses.jsonl`, existing baseline/P2 score files for the comparison
table) are read from `../results/` and `../data/` via relative paths, since the notebook's
cwd is its own folder (JupyterLab kernel behavior, see `experiment/README.md`).

## Metrics (identical methodology to `p2_structured_prompting.py`)

Per row, per condition, per reference model (`14b`/`72b`):
- `faithfulness` — RAGAS `Faithfulness`, judged by `gpt-5.5`, against the same retrieved
  clause text used for generation.
- `answer_relevancy` — RAGAS `AnswerRelevancy`, judged by `gpt-5.5` + local HF embeddings
  (`all-MiniLM-L6-v2`).
- `similarity` — cosine similarity between answer and reference embeddings (batched,
  avoiding the earlier tqdm thread-safety bug from N-at-once `aembed_text` calls).
- `llm_grade` — `gpt-5.5` yes/no substantive-correctness grade vs the reference.

## Comparison table (cell group 6)

Merges, all filtered to the same 21 queries + `hybrid` config, indexed by
`(query, variant, reference_model)`:

1. **Baseline** — `../results/correctness_scores.jsonl` + `../results/ragas_answer_scores.jsonl`
   (Llama-3.1-8B, original prompt)
2. **P2 variant 1** — `../results/p2_structured_prompting_scores.jsonl`
   (Llama-3.1-8B, completeness-instruction prompt)
3. **P2 variant 2** — `../results/p2v2_two_step_scores.jsonl`
   (Llama-3.1-8B, two-step extraction→synthesis)
4. **gptTest non-thinking** — `results/scores.jsonl` filtered to `gpt5nano_base`
5. **gptTest thinking** — `results/scores.jsonl` filtered to `gpt5nano_thinking`

Output: `results/comparison.csv` (5 variants x 21 queries x 2 reference models, all four
metrics) and `figures/comparison.png` (grouped bar chart, mean per variant per metric).

## Cell-group breakdown (built one at a time, per `plan/CLAUDE.md` convention)

1. **Setup** — imports, `.env` load, path config, load the 21 failure rows +
   `clause_lookup` + `test_set.jsonl`.
2. **API capability probe** — 1-2 row smoke test: confirm `reasoning_effort` support and
   valid values, confirm `response_format=json_object` works, confirm a working
   `max_completion_tokens` budget for both conditions, get a per-call time estimate.
3. **Generate: non-thinking** — all 21 rows → `results/answers_nonthinking.jsonl`.
4. **Generate: thinking** — all 21 rows → `results/answers_thinking.jsonl`.
5. **Score both conditions** — faithfulness/answer_relevancy/similarity/llm_grade vs both
   reference models → `results/scores.jsonl`.
6. **Merge comparison table** — join with existing baseline/P2/P2v2 scores →
   `results/comparison.csv`.
7. **Summary + plot + findings** — `figures/comparison.png`, written interpretation of
   whether either GPT-5-nano condition moves the needle on the truncation/synthesis
   failure modes, and whether thinking helps beyond the base condition.

## Change log

- 2026-07-10: Initial plan written, before any notebook cells built.
- 2026-07-10: Cell group 2 (API probe) run against the real API on 1 sample query.
  Findings, which change the "thinking" condition from what this plan originally
  specified:
  - `reasoning_effort="minimal"` works cleanly: 0 reasoning tokens, 226 completion
    tokens, 4.7s, valid JSON via `response_format={"type": "json_object"}`. Used as-is
    for the non-thinking condition.
  - `reasoning_effort="high"` does **not** converge for `gpt-5-nano` on this task at a
    practical budget: at `max_completion_tokens=4096` it consumed all 4096 as hidden
    reasoning tokens and returned empty content (`finish_reason="length"`); doubling the
    budget to 8192 reproduced the identical failure (8192/8192 reasoning tokens, still
    empty). This looks like a real capability-tier limit, not an undersized budget — the
    nano tier appears to reason indefinitely at "high" effort on this task without ever
    emitting a visible answer, unlike the judge-model empty-reply bug seen elsewhere in
    this project (which was fixed by a bigger budget, not a different effort level).
  - `reasoning_effort="medium"` was probed as a fallback and works: 1920 reasoning
    tokens, 953-char valid-JSON answer, 14.5s, `finish_reason="stop"`.
  - **Decision: the "thinking" condition uses `reasoning_effort="medium"`, not `"high"`
    as originally planned.** Both conditions now use `max_completion_tokens=8192` (kept
    equal across conditions so the token ceiling itself isn't a confound). The
    high-effort non-convergence is itself worth keeping as a small reportable finding
    (reasoning effort doesn't transfer uniformly across model tiers — the nano tier
    can't practically use its top effort setting on this task), separate from the
    thinking-vs-non-thinking comparison this experiment is actually testing.
- 2026-07-10: Full run complete — all 7 cell groups built and executed (0 generation
  errors in either condition). Results: `gpt5nano_base` (no reasoning) raises
  faithfulness over the Llama-3.1-8B baseline (0.561→0.641) but doesn't move `llm_grade`
  correctness off 0/42; `gpt5nano_thinking` (medium effort) raises faithfulness further
  (0.758) and is the first variant across the whole P1/P2/gptTest lineage to get any
  `llm_grade` passes (3/42) — but all three are `exact_anchor` queries, not
  `cross_reference`, so the gain doesn't confirm the cross-reference-synthesis
  hypothesis that motivated testing thinking in the first place. Full findings written
  up in the notebook's final markdown cell and in `plan/report.md`'s 2026-07-10 "gptTest"
  entry; `plan/STATUS.md` updated. Outputs: `results/answers_nonthinking.jsonl`,
  `results/answers_thinking.jsonl`, `results/scores.jsonl`, `results/comparison.csv`,
  `figures/comparison.png`.
- 2026-07-10: **Extended to all four retrieval configs** (cell groups 8-10), at the
  user's request, reusing each config's already-recorded `retrieved` clause IDs from
  `p1_universal_failures.jsonl` — no new retrieval needed. 126 new generation calls (3
  configs x 2 conditions x 21 queries, 0 errors) + scoring against both reference
  models. Key revision to the hybrid-only reading above: the faithfulness gain from
  swapping to GPT-5-nano holds in every config (baseline 0.440-0.601 range; both GPT
  conditions beat it everywhere), but the `llm_grade` correctness gain does **not**
  cleanly track reasoning effort the way the hybrid-only result suggested —
  `sparse_only` gets the same-sized gain (3/42) already at the non-thinking condition,
  while `hybrid` needed thinking to reach it; `dense_only`/`dense_sparse` show a smaller,
  reasoning-effort-invariant gain (1/42 at both levels). Revised read: the correctness
  gain is real but small, and "thinking is what unlocks it" doesn't hold as a general
  claim — different configs reach their (still small) ceiling by different routes.
  Outputs: `results/answers_{nonthinking,thinking}_{dense_only,sparse_only,dense_sparse}.jsonl`,
  `results/scores_extra_configs.jsonl`, `results/comparison_all_configs.csv`,
  `figures/comparison_all_configs.png`.
