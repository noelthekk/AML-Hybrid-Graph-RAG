"""P1: extract the universal-failure queries (llm_grade=0 across all four configs, both
reference models, at BOTH generator scales) with full context for manual categorization.

See plan/improvement_plan.md P1 for the full design. This script does the automatable
part (finding the queries, checking whether the gold clause was ever retrieved); the
categorization (retrieval gap / generation synthesis failure / test-set ambiguity /
other) is a manual read of the printed output, not scripted.

Usage:
    uv run python experiment/p1_failure_analysis.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def universal_failures(correctness_rows: list[dict], config_key: str = "config") -> set[str]:
    """Queries where llm_grade==0 for every (config, reference_model) combination."""
    by_query: dict[str, dict[str, list[float]]] = {}
    for row in correctness_rows:
        if row.get("llm_grade") is None:
            continue
        by_query.setdefault(row["query"], {}).setdefault(row[config_key], []).append(row["llm_grade"])

    all_configs = {row[config_key] for row in correctness_rows}
    fails = set()
    for query, by_config in by_query.items():
        if set(by_config.keys()) != all_configs:
            continue  # incomplete data for this query, skip
        if all(all(g == 0 for g in grades) for grades in by_config.values()):
            fails.add(query)
    return fails


def main() -> None:
    correctness_8b = load_jsonl(RESULTS_DIR / "correctness_scores.jsonl")
    correctness_70b = load_jsonl(RESULTS_DIR / "ragas_correctness_scores_70b.jsonl")

    # 8B rows use "ref_model", 70B rows use "reference_model" - normalize
    for r in correctness_8b:
        r["reference_model"] = r.get("ref_model", r.get("reference_model"))

    fail_8b = universal_failures(correctness_8b)
    fail_70b = universal_failures(correctness_70b)
    fail_both = fail_8b & fail_70b

    print(f"8B universal failures: {len(fail_8b)}")
    print(f"70B universal failures: {len(fail_70b)}")
    print(f"Fail at BOTH generator scales: {len(fail_both)}")
    print()

    answers_8b = load_jsonl(RESULTS_DIR / "answers_recovered.jsonl")
    answers_70b = load_jsonl(RESULTS_DIR / "answers_70b_recovered.jsonl")
    test_set = load_jsonl(Path("data/test_set.jsonl"))
    ts_lookup = {r["query"]: r for r in test_set}

    a8b_by_query: dict[str, list[dict]] = {}
    for r in answers_8b:
        a8b_by_query.setdefault(r["query"], []).append(r)
    a70b_by_query: dict[str, list[dict]] = {}
    for r in answers_70b:
        a70b_by_query.setdefault(r["query"], []).append(r)

    out_path = RESULTS_DIR / "p1_universal_failures.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, query in enumerate(sorted(fail_both), start=1):
            ts_row = ts_lookup[query]
            gold_ids = set(ts_row["gold_ids"])

            rows_8b = a8b_by_query.get(query, [])
            rows_70b = a70b_by_query.get(query, [])

            # Was the gold clause ever retrieved, by any config, at either generator scale?
            # (retrieval doesn't depend on generator, so 8B/70B rows for the same
            # config should have identical `retrieved` lists - checking both anyway
            # as a sanity cross-check, not assuming.)
            retrieved_hit = {}
            for r in rows_8b + rows_70b:
                cfg = r["config"]
                hit = bool(gold_ids & set(r["retrieved"]))
                retrieved_hit.setdefault(cfg, []).append(hit)
            any_config_retrieved_gold = any(any(hits) for hits in retrieved_hit.values())

            record = {
                "n": i,
                "query": query,
                "query_type": ts_row["query_type"],
                "gold_ids": ts_row["gold_ids"],
                "gold_ever_retrieved": any_config_retrieved_gold,
                "retrieved_hit_by_config": retrieved_hit,
                "reference_72b": ts_row["reference_72b"],
                "answers_8b": {r["config"]: {"answer": r["answer"], "citations": r["citations"],
                                              "retrieved": r["retrieved"]} for r in rows_8b},
                "answers_70b": {r["config"]: {"answer": r["answer"], "citations": r["citations"]}
                                for r in rows_70b},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(fail_both)} rows to {out_path}")
    n_retrieval_gap = sum(
        1 for l in out_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not json.loads(l)["gold_ever_retrieved"]
    )
    print(f"Quick automated split: {n_retrieval_gap}/{len(fail_both)} never had the gold "
          f"clause retrieved by ANY config - {len(fail_both) - n_retrieval_gap}/{len(fail_both)} "
          f"had it retrieved somewhere but still failed (needs manual read to see why).")


if __name__ == "__main__":
    main()
