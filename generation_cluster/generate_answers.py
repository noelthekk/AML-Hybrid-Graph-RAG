"""Generate answers for the RAGAS test set across all four retrieval configs.

Run this on the cluster. Generator is Llama-3.1-8B-Instruct, BF16 - fixed regardless of
hardware (see plan.md Sec 5, "Generator model identity"): a controlled comparison across
retrieval configs needs the same generator throughout, so this deliberately does not
offer a bigger-model option the way the reference-drafting cluster/ folder does. Only the
precision differs from the local notebook (BF16 here vs NF4 4-bit locally) - never mind
which one is more convenient, using a different generator identity would confound
retrieval-quality differences with generator-quality differences.

Self-contained: no dependency on the rest of the aml-hybrid-rag project. Includes its own
copies of scripts/retriever.py, scripts/graph_build.py, and the data files they need
(clauses.jsonl, cross_refs.jsonl, chroma_db/) plus test_set.jsonl.

Before running:
    1. Copy .env.example to .env and set HF_TOKEN.
    2. uv sync

Usage:
    uv run python generate_answers.py

Output: results/answers.jsonl - one row per (config, query) pair, 200 rows total (4
configs x 50 queries): query, gold_ids, query_type, config, retrieved (clause_id list),
answer, citations. Rows where the model's JSON output didn't parse get an `answer` field
starting with "Generation error:" and an empty `citations` list - check for those first.
"""
import json
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline_fn
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from retriever import (  # noqa: E402
    load_retrievers, dense_only_retrieve, sparse_only_retrieve,
    dense_sparse_retrieve, hybrid_retrieve,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_answers")

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TEST_SET_PATH = DATA_DIR / "test_set.jsonl"
OUTPUT_PATH = RESULTS_DIR / "answers.jsonl"

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # fixed regardless of hardware -
                                                       # see plan.md Sec 5, "Generator
                                                       # model identity"
MAX_NEW_TOKENS = 1024
TOP_K = 10
RRF_K = 60
GRAPH_HOPS = 2

ALL_CONFIGS = ["dense_only", "sparse_only", "dense_sparse", "hybrid"]

SYSTEM_PROMPT = """You are an AML compliance analyst. Answer using ONLY the regulatory clauses below.

Rules:
1. Every factual claim must be followed by [clause_id] inline.
2. State explicitly if the provided context does not answer the question.
3. Output valid JSON with two keys:
   - answer: your response with inline [clause_id] citations
   - citations: list of clause_id strings you cited

Context:
{context}"""


def format_context(results: list) -> str:
    parts = []
    for r in results:
        cid = r["clause_id"]
        text = r["text"][:800]
        parts.append(f"[{cid}]\n{text}")
    return "\n\n---\n\n".join(parts)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve_for_config(config: str, query: str, vectorstore, bm25, clauses, G) -> list:
    if config == "dense_only":
        return dense_only_retrieve(vectorstore, clauses, query, k=TOP_K)
    if config == "sparse_only":
        return sparse_only_retrieve(bm25, clauses, query, k=TOP_K)
    if config == "dense_sparse":
        return dense_sparse_retrieve(query, vectorstore, bm25, clauses, k=TOP_K, rrf_k=RRF_K)
    if config == "hybrid":
        return hybrid_retrieve(query, vectorstore, bm25, clauses, G, k=TOP_K, graph_hops=GRAPH_HOPS, rrf_k=RRF_K)
    raise ValueError(f"unknown config: {config}")


def main() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA device found - this script requires a GPU.")

    logger.info("Loading retrievers from %s", DATA_DIR)
    t0 = time.time()
    vectorstore, bm25, clauses, G = load_retrievers(DATA_DIR, CHROMA_DIR)
    logger.info("Retrievers loaded: %d clauses  (%.1fs)", len(clauses), time.time() - t0)

    test_set = load_jsonl(TEST_SET_PATH)
    logger.info("Loaded %d test queries", len(test_set))

    logger.info("Loading %s (BF16)...", MODEL_ID)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    gen_pipe = hf_pipeline_fn(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, return_full_text=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipe))
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    chain = prompt | llm | parser
    logger.info("Model loaded (%.1fs)  GPU: %s", time.time() - t0, torch.cuda.get_device_name(0))

    rows = []
    t0 = time.time()
    n_total = len(ALL_CONFIGS) * len(test_set)
    n_done = 0
    n_errors = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for config in ALL_CONFIGS:
            for item in test_set:
                query = item["query"]
                retrieved = retrieve_for_config(config, query, vectorstore, bm25, clauses, G)
                context = format_context(retrieved)
                try:
                    parsed = chain.invoke({"query": query, "context": context})
                    answer = parsed.get("answer", "")
                    citations = parsed.get("citations", [])
                except Exception as exc:
                    answer = f"Generation error: {exc}"
                    citations = []
                    n_errors += 1

                row = {
                    "query": query,
                    "gold_ids": item["gold_ids"],
                    "query_type": item.get("query_type", "unknown"),
                    "config": config,
                    "retrieved": [r["clause_id"] for r in retrieved],
                    "answer": answer,
                    "citations": citations,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                n_done += 1
                elapsed = time.time() - t0
                avg = elapsed / n_done
                remaining = (n_total - n_done) * avg
                logger.info(
                    "[%d/%d] config=%s  %r -> %d chars"
                    "  (avg %.1fs/query, ~%.0fs remaining)",
                    n_done, n_total, config, query[:50], len(answer), avg, remaining,
                )

    logger.info(
        "Done: %d rows written to %s  (%d generation errors)",
        len(rows), OUTPUT_PATH, n_errors,
    )


if __name__ == "__main__":
    main()
