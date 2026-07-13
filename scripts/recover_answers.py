"""Recover answers from strict-JSON parse failures in a generated answers file.

Model output is parsed with langchain's strict JsonOutputParser; when a model wraps its
JSON in prose (or skips JSON entirely), the row is stored as
`answer = "Generation error: Invalid json output: <raw output>"`. Decoding is
greedy/deterministic, so re-generating reproduces the same text - re-parsing the
existing raw output is the correct fix, not re-running. Matters most for 8B-vs-70B
comparisons: the 70B wraps JSON in prose far more often (145/200 vs 3/200), so without
recovery the comparison would measure JSON-formatting compliance more than answer
quality. Citations are extracted unfiltered in both the JSON and prose-fallback paths,
so neither gets an artificial citation-validity advantage. Recovered rows gain a
`recovered` field (`json`/`prose`); non-error rows pass through unchanged; the input
file itself is never modified.

Usage:
    python scripts/recover_answers.py results/answers.jsonl results/answers_recovered.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ERROR_PREFIX = "Generation error: Invalid json output: "
# langchain appends this hint to OutputParserException messages - not model output
TROUBLESHOOT_RE = re.compile(r"\s*For troubleshooting, visit: \S+\s*$")
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def try_json(raw: str) -> dict | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "answer" not in parsed:
        return None
    citations = parsed.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    return {"answer": str(parsed["answer"]), "citations": [str(c) for c in citations]}


def prose_citations(raw: str) -> list[str]:
    seen: list[str] = []
    for match in BRACKET_RE.findall(raw):
        for token in match.split(","):
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
    return seen


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument("input", type=Path)
    arg_parser.add_argument("output", type=Path)
    args = arg_parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        sys.exit("input and output must differ - the input file is never modified")

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    n_json = n_prose = n_unrecovered = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            answer = row["answer"]
            if answer.startswith(ERROR_PREFIX):
                raw = TROUBLESHOOT_RE.sub("", answer[len(ERROR_PREFIX):]).strip()
                parsed = try_json(raw)
                if parsed is not None:
                    row = {**row, **parsed, "recovered": "json"}
                    n_json += 1
                elif raw:
                    row = {**row, "answer": raw,
                           "citations": prose_citations(raw), "recovered": "prose"}
                    n_prose += 1
                else:
                    n_unrecovered += 1
            elif answer.startswith("Generation error:"):
                n_unrecovered += 1  # a different error type - leave untouched
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_err = n_json + n_prose + n_unrecovered
    print(f"{args.input} -> {args.output}")
    print(f"rows: {len(rows)}, error rows: {total_err}")
    print(f"recovered via JSON extraction: {n_json}")
    print(f"recovered as prose (citations from inline brackets): {n_prose}")
    print(f"unrecovered (left as-is): {n_unrecovered}")


if __name__ == "__main__":
    main()
