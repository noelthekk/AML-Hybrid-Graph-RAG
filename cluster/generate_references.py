"""Draft reference answers for the RAGAS test set using Qwen2.5-Instruct.

Run this on the cluster. Model size and required VRAM partition are chosen with
--model-size, since they determine which partition to request:
    14b -> Qwen2.5-14B-Instruct, NF4 4-bit, ~7-8GB weights  -> use an 18GB partition
    72b -> Qwen2.5-72B-Instruct, NF4 4-bit, ~36-40GB weights -> use a 72GB partition
(72b was always the originally-preferred size - see plan.md Sec 5 - with 14b as the
fallback for when only 18GB is accessible. No default: get this wrong and you either
OOM on 18GB or waste headroom on 72GB, so it has to be a conscious choice each run.)

Self-contained: no dependency on the rest of the aml-hybrid-rag project.

Before running:
    1. Copy data/test_set.jsonl and data/clauses.jsonl from the main project's
       plan/implementation/data/ into this project's data/ (already done if this
       folder was copied as-is from the main project).
    2. Copy .env.example to .env and set HF_TOKEN.
    3. uv sync

Usage:
    uv run python generate_references.py --model-size 14b
    uv run python generate_references.py --model-size 72b

Progress logs to the terminal with timestamps as it goes.

Output: results/reference_drafts_<model-size>.jsonl (e.g. reference_drafts_72b.jsonl) -
one row per test-set query, with the drafted `reference` and self-reported `citations`,
plus an automated pre-check against `gold_ids` logged at the end. The model-size suffix
keeps a 14b run and a 72b run from overwriting each other if you try both.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_references")
# StreamHandler flushes after every record by default, unlike print() to a pipe - this
# is what avoids the delayed/batched output seen from a plain-print script earlier in
# this project.

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SET_PATH = DATA_DIR / "test_set.jsonl"
CLAUSES_PATH = DATA_DIR / "clauses.jsonl"

MODEL_IDS = {
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}
MAX_NEW_TOKENS = 768

SYSTEM_PROMPT = """You are drafting a reference answer for a UK AML compliance test set. \
You will be given a question and the exact clause text that grounds the answer. \
Answer using only the information in the clause text - do not add outside knowledge. \
If the question has multiple parts, answer every part.

Respond with a JSON object with exactly two fields:
  "reference": a short, correct answer in plain prose (a few sentences, no bracket citations)
  "citations": a list of the clause IDs (shown in brackets in the Clauses section below) \
that your answer actually relies on

Respond with JSON only, no other text, no markdown code fences."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(query: str, gold_ids: list[str], clause_text: dict[str, str]) -> str:
    clauses_block = "\n\n".join(
        f"[{cid}]\n{clause_text[cid]}" for cid in gold_ids if cid in clause_text
    )
    return f"Clauses:\n{clauses_block}\n\nQuestion: {query}"


def parse_response(text: str) -> dict:
    """Best-effort JSON extraction - local models sometimes wrap JSON in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"reference": text, "citations": [], "_parse_error": "no JSON object found"}
    try:
        parsed = json.loads(text[start:end + 1])
        return {
            "reference": parsed.get("reference", ""),
            "citations": parsed.get("citations", []),
        }
    except json.JSONDecodeError as e:
        return {"reference": text, "citations": [], "_parse_error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-size",
        required=True,
        choices=sorted(MODEL_IDS),
        help="14b needs an 18GB partition, 72b needs a 72GB partition (see plan.md Sec 5)",
    )
    args = parser.parse_args()
    model_id = MODEL_IDS[args.model_size]
    output_path = RESULTS_DIR / f"reference_drafts_{args.model_size}.jsonl"

    logger.info("Model size: %s (%s)", args.model_size, model_id)
    logger.info("Output: %s", output_path)

    test_set = load_jsonl(TEST_SET_PATH)
    clauses = load_jsonl(CLAUSES_PATH)
    clause_text = {c["clause_id"]: c["text"] for c in clauses}
    logger.info("Loaded %d test-set rows, %d clauses", len(test_set), len(clauses))

    logger.info("Loading %s (NF4 4-bit)...", model_id)
    t0 = time.time()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto"
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    logger.info("Model loaded (%.1fs)", time.time() - t0)

    rows = []
    t0 = time.time()
    for i, item in enumerate(test_set):
        query = item["query"]
        gold_ids = item["gold_ids"]
        user_prompt = build_prompt(query, gold_ids, clause_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw_output = generator(prompt_text)[0]["generated_text"]
        parsed = parse_response(raw_output)

        row = {**item, "draft_reference": parsed["reference"], "citations": parsed["citations"]}
        if "_parse_error" in parsed:
            row["_parse_error"] = parsed["_parse_error"]
        rows.append(row)

        progress = (
            f"[{i + 1}/{len(test_set)}] {query[:60]!r} -> "
            f"{len(parsed['reference'])} chars, {len(parsed['citations'])} citations"
        )
        if "_parse_error" in parsed:
            logger.warning("%s  [PARSE ERROR: %s]", progress, parsed["_parse_error"])
        else:
            logger.info(progress)

    elapsed = time.time() - t0
    logger.info("Drafted %d rows in %.1fs", len(rows), elapsed)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved to %s", output_path)

    # Automated pre-check (matches Phase A task #17 in the main project's plan):
    # citations should match gold_ids as a set.
    logger.info("--- Automated pre-check: citations vs gold_ids ---")
    n_flagged = 0
    for row in rows:
        if set(row["citations"]) != set(row["gold_ids"]):
            n_flagged += 1
            missing = set(row["gold_ids"]) - set(row["citations"])
            extra = set(row["citations"]) - set(row["gold_ids"])
            logger.warning("FLAGGED: %r", row["query"][:60])
            if missing:
                logger.warning("  missing from citations: %s", sorted(missing))
            if extra:
                logger.warning("  extra in citations (not in gold_ids): %s", sorted(extra))
    if n_flagged:
        logger.warning(
            "%d/%d rows flagged - review these first during human verification.",
            n_flagged, len(rows),
        )
    else:
        logger.info("All %d rows: citations match gold_ids exactly.", len(rows))


if __name__ == "__main__":
    main()
