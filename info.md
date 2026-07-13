# Generator Comparison: Llama-3.1-8B vs GPT-5-nano

Comparison of five generation setups against the 21 hardest queries in the test set
(queries that scored 0 on correctness across every retrieval config, both reference
models, and both generator scales in the original 200-query evaluation). Retrieval,
reference answers, and the scoring judge are held identical across every setup, so all
results below are directly comparable.

## Setups tried

| Setup | Generator | Change from baseline | Configs run |
|---|---|---|---|
| Llama-3.1-8B (original prompt) | Llama-3.1-8B-Instruct, local | None (baseline) | all 4 |
| P2v1 (structured prompting) | Llama-3.1-8B-Instruct, local | Prompt rule: state every condition a clause lists, not just the first | hybrid |
| P2v2 (two-step extraction) | Llama-3.1-8B-Instruct, local | Split generation into per-clause extraction + synthesis | hybrid |
| gpt5nano_base | GPT-5-nano, OpenAI API | Different generator, no reasoning | all 4 |
| gpt5nano_thinking | GPT-5-nano, OpenAI API | Different generator, medium reasoning effort | all 4 |

Retrieval configs: `dense_only`, `sparse_only`, `dense_sparse`, `hybrid` (graph-expanded).

## Metrics tracked

| Metric | What it measures | Depends on generator? |
|---|---|---|
| `context_recall` | Does retrieved context cover what's needed | No (retrieval-side) |
| `context_precision` | How much retrieved context is relevant | No (retrieval-side) |
| Faithfulness | Are the answer's claims grounded in retrieved context | Yes |
| Answer relevancy | Is the answer relevant to the question | Yes |
| Similarity | Cosine similarity to the reference answer | Yes |
| `llm_grade` | Binary correctness vs. reference (0 or 1), judged by an LLM | Yes |

## Why each fix was tried, and what happened

All four fixes below target the same problem: Llama-3.1-8B correctly identifies the
right clause but stops after the first condition or two, dropping the rest.

- `P2v1` (prompt fix): worked mechanically (answers got more complete) but barely
  moved correctness (1/42 vs. 0/42 baseline), and caused one answer to loop
  repetitively.
- `P2v2` (structural fix): fixed the repetition problem, but faithfulness and
  relevancy both dropped below baseline, and correctness returned to 0/42. New
  failure modes: paraphrase drift, and over-compressed final answers.
- `gpt5nano_base` (different generator): raised faithfulness in every config, but
  only moved correctness in `sparse_only` (3/42); other configs saw a smaller gain
  (1/42) or none (`hybrid`, 0/42).
- `gpt5nano_thinking` (added reasoning): the first fix to raise `hybrid`'s
  correctness (3/42), but the gain came from `exact_anchor` queries, not the
  `cross_reference` queries reasoning was meant to help. `sparse_only` reached the
  same 3/42 ceiling without any reasoning at all.

No fix beat 3/42 (7%) correctness on this hardest-query subset, and each introduced its
own new trade-off rather than solving the underlying truncation problem.

## Results

| Setup | Config | context_recall | context_precision | Faithfulness | Answer relevancy | Similarity | `llm_grade` |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | dense_only | 0.531 | 0.421 | 0.440 | 0.740 | 0.799 | 0/42 |
| Llama-3.1-8B | sparse_only | 0.588 | **0.494** | 0.601 | 0.792 | 0.811 | 0/42 |
| Llama-3.1-8B | dense_sparse | 0.557 | 0.418 | 0.527 | 0.785 | **0.825** | 0/42 |
| Llama-3.1-8B | hybrid | **0.592** | 0.435 | 0.561 | 0.798 | 0.819 | 0/42 |
| P2v1 | hybrid | **0.592** | 0.435 | 0.570 | 0.738 | 0.796 | 1/42 |
| P2v2 | hybrid | **0.592** | 0.435 | 0.502 | 0.707 | 0.780 | 0/42 |
| gpt5nano_base | dense_only | 0.531 | 0.421 | 0.577 | **0.800** | 0.802 | 1/42 |
| gpt5nano_base | sparse_only | 0.588 | **0.494** | 0.616 | 0.672 | 0.807 | **3/42** |
| gpt5nano_base | dense_sparse | 0.557 | 0.418 | 0.557 | 0.747 | 0.802 | 1/42 |
| gpt5nano_base | hybrid | **0.592** | 0.435 | 0.641 | 0.694 | 0.791 | 0/42 |
| gpt5nano_thinking | dense_only | 0.531 | 0.421 | 0.739 | 0.682 | 0.744 | 1/42 |
| gpt5nano_thinking | sparse_only | 0.588 | **0.494** | 0.717 | 0.567 | 0.715 | **3/42** |
| gpt5nano_thinking | dense_sparse | 0.557 | 0.418 | 0.755 | 0.608 | 0.713 | 1/42 |
| gpt5nano_thinking | hybrid | **0.592** | 0.435 | **0.758** | 0.673 | 0.768 | **3/42** |

Bold marks the best value in each metric column. `context_recall`/`context_precision`
repeat per config since they don't depend on the generator.

## Key takeaways

- Best faithfulness: `gpt5nano_thinking` + `hybrid` (0.758).
- Best correctness: three-way tie at 3/42, `gpt5nano_base`/`gpt5nano_thinking` on
  `sparse_only`, and `gpt5nano_thinking` on `hybrid`. `sparse_only` reaches this for
  free; `hybrid` needs reasoning to match it.
- Best answer relevancy/similarity: always Llama-3.1-8B, in every config. GPT-5-nano
  trades relevancy/similarity for faithfulness, more so with reasoning enabled.
- Worst overall: P2v2, lowest faithfulness and relevancy of all 14 setups.
- Even `hybrid`'s retrieval recall (0.592) is well below 1.0 on this hardest-query
  subset, so part of the correctness ceiling may be a retrieval limitation, not only a
  generation one.

Full detail (including why `reasoning_effort="high"` was rejected in favor of
`"medium"`): `gptTest/plan.md`. Report-ready write-up: `../report.md`'s "gptTest" entry,
2026-07-10.
