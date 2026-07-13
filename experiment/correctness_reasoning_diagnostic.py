"""Diagnostic: why does the correctness judge grade some high-faithfulness /
high-similarity rows as incorrect, with no recorded reasoning?

Flagged in report.md's 2026-07-09 (P1 #21) and 2026-07-10 (P2 variant 1 #3, variant 2
multiple rows) entries: several rows score faithfulness 1.00 and similarity 0.80-0.94
yet still get a flat "no" from the yes/no correctness judge. This re-runs the judge on
a handful of those specific rows with a modified prompt that asks for a one-sentence
reasoning field before the verdict, so the actual disagreement is visible instead of
just a verdict.

Not a new experiment variant - a one-off diagnostic. Selected rows are hardcoded below
based on faithfulness/similarity/grade values already found in this session's analysis.

Run: uv run python experiment/correctness_reasoning_diagnostic.py
"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI

RESULTS_DIR = Path("results")
JUDGE_MODEL = "gpt-5.5"

REASONING_CORRECTNESS_PROMPT = """You are grading whether a candidate answer is substantively \
correct compared to a reference answer, for a UK AML compliance question-answering system. \
Judge only substantive correctness (facts, conditions, obligations stated) - ignore \
differences in phrasing, style, or which clause IDs are cited inline.

Respond with valid JSON with two keys:
- reasoning: one or two sentences explaining what, if anything, the candidate answer is
  missing or gets wrong relative to the reference - be specific about the exact gap
- verdict: "yes" if the candidate answer is substantively correct and complete relative
  to the reference, or "no" if it is incorrect, incomplete, or contradicts the
  reference"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


async def graded_reasoning(client, query, reference, answer):
    user_prompt = f"Question: {query}\n\nReference answer: {reference}\n\nCandidate answer: {answer}\n\nIs the candidate answer substantively correct and complete relative to the reference?"
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": REASONING_CORRECTNESS_PROMPT},
                  {"role": "user", "content": user_prompt}],
        max_completion_tokens=4096, temperature=1.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


async def main():
    client = AsyncOpenAI()
    fails = load_jsonl(RESULTS_DIR / "p1_universal_failures.jsonl")
    fails_by_query = {r["query"]: r for r in fails}
    test_set = load_jsonl(Path("data/test_set.jsonl"))
    ts_lookup = {r["query"]: r for r in test_set}
    v1_answers = {r["query"]: r for r in load_jsonl(RESULTS_DIR / "p2_structured_prompting_answers.jsonl")}
    v2_answers = {r["query"]: r for r in load_jsonl(RESULTS_DIR / "p2v2_two_step_answers.jsonl")}

    def find_query(prefix):
        matches = [q for q in fails_by_query if q.startswith(prefix)]
        assert len(matches) == 1, f"expected 1 match for {prefix!r}, got {len(matches)}"
        return matches[0]

    Q_REG29 = find_query("MLR 2017 regulation 29 specifies")
    Q_REG30 = find_query("Regulation 30 of MLR 2017 sets")
    Q_S331 = find_query("Under section 331 of POCA 2002")

    # picked for high similarity/faithfulness but llm_grade == 0 against reference_72b,
    # confirmed against the actual scores files rather than guessed - see report.md's
    # 2026-07-10 correctness-judge-reasoning diagnostic entry
    cases = [
        ("reg29, baseline (sim=0.98, faith=0.75)", Q_REG29, "baseline", "72b"),
        ("reg29, variant 1 (sim=0.88, faith=1.00)", Q_REG29, "v1", "72b"),
        ("reg29, variant 2 (sim=0.93, faith=1.00)", Q_REG29, "v2", "72b"),
        ("reg30, variant 1 (sim=0.94, faith=1.00)", Q_REG30, "v1", "72b"),
        ("s331, baseline (sim=0.93, faith=0.78)", Q_S331, "baseline", "72b"),
    ]

    tasks = []
    labels = []
    for label, query, source, ref_model in cases:
        reference = ts_lookup[query][f"reference_{ref_model}"]
        if source == "baseline":
            answer = fails_by_query[query]["answers_8b"]["hybrid"]["answer"]
        elif source == "v1":
            answer = v1_answers[query]["answer"]
        else:
            answer = v2_answers[query]["answer"]
        labels.append((label, query, reference, answer))
        tasks.append(graded_reasoning(client, query, reference, answer))

    results = await asyncio.gather(*tasks)

    for (label, query, reference, answer), result in zip(labels, results):
        print(f"=== {label} ===")
        print(f"QUERY: {query}")
        print(f"REFERENCE: {reference}")
        print(f"CANDIDATE: {answer}")
        print(f"VERDICT: {result.get('verdict')}")
        print(f"REASONING: {result.get('reasoning')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
