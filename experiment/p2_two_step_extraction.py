"""P2 variant 2: two-step per-clause extraction, then synthesis.

Variant 1 (p2_structured_prompting.py) told the single generation call to "state
every condition, don't stop after the first" - it worked mechanically (answers got
more complete) but didn't move llm_grade correctness, and on one row it removed the
model's natural stopping signal badly enough to cause a greedy-decoding repetition
loop (see plan/report.md's 2026-07-09 entry).

Variant 2 splits the task instead of instructing harder on the same task:
  Step A (extraction, one short call per retrieved clause): "list every condition,
  requirement, or element THIS clause states relevant to the question, don't omit
  any" - a small, constrained task, less prone to open-ended looping.
  Step B (synthesis, one call per query): combine the already-complete per-clause
  extractions into a cited answer. The synthesis model isn't asked to dig
  completeness out of raw legal text itself - that's already done for it.

Same 21 universal-failure queries, same `hybrid` retrieved clause IDs (reused from
results/answers_recovered.jsonl via p1's output, not re-run), same local 8B
generator, same RAGAS scoring pipeline as variant 1 - only the generation strategy
differs, isolating that as the variable.

Three phases, run separately:
    uv run python experiment/p2_two_step_extraction.py extract
    uv run python experiment/p2_two_step_extraction.py synthesize
    uv run python experiment/p2_two_step_extraction.py score
"""
import asyncio
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

RESULTS_DIR = Path("results")
EXTRACTIONS_PATH = RESULTS_DIR / "p2v2_extractions.jsonl"
ANSWERS_PATH = RESULTS_DIR / "p2v2_two_step_answers.jsonl"
SCORES_PATH = RESULTS_DIR / "p2v2_two_step_scores.jsonl"

EXTRACTION_SYSTEM_PROMPT = """You are extracting information from a single regulatory clause to help answer a compliance question.

Given the clause text below and the question, list every condition, requirement, or element the clause states that is relevant to the question, as a plain bullet list (one bullet per condition/requirement/element). Do not omit any, including the clause's concluding or operative sentence. If the clause is not relevant to the question, reply with exactly: NOT RELEVANT."""

SYNTHESIS_SYSTEM_PROMPT = """You are an AML compliance analyst. Below, for each retrieved regulatory clause, is a complete pre-extracted list of every condition, requirement, or element that clause states relevant to the question - already fully extracted, so you do not need to re-read or trim the original clause text yourself.

Rules:
1. Every factual claim must be followed by [clause_id] inline.
2. State explicitly if the provided extractions do not answer the question.
3. Output valid JSON with two keys:
   - answer: your response with inline [clause_id] citations
   - citations: list of clause_id strings you cited
4. Combine ALL relevant extracted conditions, requirements, or elements into your
   answer - each clause's extraction below has already been checked for
   completeness, so include every item listed, not just the first item per clause.

Extracted clause summaries:
{context}"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract() -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import pipeline as hf_pipeline_fn

    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA device found.")

    failures = load_jsonl(RESULTS_DIR / "p1_universal_failures.jsonl")
    clauses = load_jsonl(Path("data/clauses.jsonl"))
    clause_lookup = {c["clause_id"]: c for c in clauses}
    print(f"Extracting per-clause conditions for {len(failures)} queries "
          f"x {len(failures[0]['answers_8b']['hybrid']['retrieved'])} retrieved clauses each")

    MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map={"": 0})
    gen_pipe = hf_pipeline_fn(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=300, do_sample=False, return_full_text=False,
    )
    print(f"Model loaded ({time.time()-t0:.1f}s)")

    def build_prompt(query: str, clause_id: str, clause_text: str) -> str:
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nClause [{clause_id}]:\n{clause_text[:800]}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Sequential, not batched: this GPU is a tight 6GB local slice already ~95% full
    # with just the model loaded, and a first attempt at batch_size=8 batching sat at
    # 98% GPU util with no output for 9+ minutes (likely VRAM-pressure thrashing, not
    # genuine speedup) - killed and reverted to the single-item-call pattern already
    # proven reliable in p2_structured_prompting.py on this same card.
    with EXTRACTIONS_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(failures, start=1):
            query = row["query"]
            retrieved_ids = row["answers_8b"]["hybrid"]["retrieved"]
            valid_ids, extractions = [], []
            t_q = time.time()
            for cid in retrieved_ids:
                clause = clause_lookup.get(cid)
                if not clause:
                    continue
                prompt = build_prompt(query, cid, clause["text"])
                out = gen_pipe(prompt)
                extractions.append(out[0]["generated_text"].strip())
                valid_ids.append(cid)

            out_row = {"query": query, "retrieved": valid_ids, "extractions": extractions}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            n_relevant = sum(1 for e in extractions if e.strip().upper() != "NOT RELEVANT")
            print(f"[{i}/{len(failures)}] {query[:50]!r} -> {n_relevant}/{len(valid_ids)} relevant  ({time.time()-t_q:.1f}s)")

    print(f"\nDone: {len(failures)} rows written to {EXTRACTIONS_PATH}")


def synthesize() -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import pipeline as hf_pipeline_fn
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA device found.")

    failures = load_jsonl(RESULTS_DIR / "p1_universal_failures.jsonl")
    extractions_by_query = {r["query"]: r for r in load_jsonl(EXTRACTIONS_PATH)}
    print(f"Synthesizing final answers for {len(failures)} queries")

    MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map={"": 0})
    gen_pipe = hf_pipeline_fn(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=1024, do_sample=False, return_full_text=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipe))
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([("system", SYNTHESIS_SYSTEM_PROMPT), ("human", "{query}")])
    chain = prompt | llm | parser
    print(f"Model loaded ({time.time()-t0:.1f}s)")

    n_errors = 0
    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(failures, start=1):
            query = row["query"]
            extraction_row = extractions_by_query[query]
            parts = []
            for cid, extraction in zip(extraction_row["retrieved"], extraction_row["extractions"]):
                if extraction.strip().upper() != "NOT RELEVANT":
                    parts.append(f"[{cid}]\n{extraction}")
            context = "\n\n---\n\n".join(parts)

            t_q = time.time()
            try:
                parsed = chain.invoke({"query": query, "context": context})
                answer = parsed.get("answer", "")
                citations = parsed.get("citations", [])
            except Exception as exc:
                answer = f"Generation error: {exc}"
                citations = []
                n_errors += 1

            out = {
                "query": query, "query_type": row["query_type"], "gold_ids": row["gold_ids"],
                "config": "hybrid", "retrieved": extraction_row["retrieved"],
                "answer": answer, "citations": citations,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(failures)}] {query[:50]!r} -> {len(answer)} chars  ({time.time()-t_q:.1f}s)")

    print(f"\nDone: {len(failures)} rows written to {ANSWERS_PATH}  ({n_errors} generation errors)")


async def _score_async() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    import numpy as np
    from openai import AsyncOpenAI

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = _stub

    from ragas.llms import llm_factory
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings as RagasHFEmbeddings
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy

    JUDGE_MODEL = "gpt-5.5"
    async_client = AsyncOpenAI()
    judge = llm_factory(JUDGE_MODEL, client=async_client)
    judge.model_args["max_completion_tokens"] = 4096
    judge.model_args["temperature"] = 1.0
    judge.model_args.pop("top_p", None)
    judge.model_args.pop("max_tokens", None)

    embeddings = RagasHFEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", use_api=False)
    metric_faithfulness = Faithfulness(llm=judge)
    metric_answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
    print("Judge + embeddings + metrics ready")

    CORRECTNESS_SYSTEM_PROMPT = """You are grading whether a candidate answer is substantively \
correct compared to a reference answer, for a UK AML compliance question-answering system. \
Judge only substantive correctness (facts, conditions, obligations stated) - ignore \
differences in phrasing, style, or which clause IDs are cited inline.

Respond with exactly one word: "yes" if the candidate answer is substantively correct and \
complete relative to the reference, or "no" if it is incorrect, incomplete, or contradicts \
the reference."""

    async def cosine_similarities_batch(texts_a, texts_b):
        all_vecs = np.array(await embeddings.aembed_texts(texts_a + texts_b))
        n = len(texts_a)
        va, vb = all_vecs[:n], all_vecs[n:]
        return (va * vb).sum(axis=1).tolist()

    async def llm_correctness_grade(query, reference, answer):
        user_prompt = f"Question: {query}\n\nReference answer: {reference}\n\nCandidate answer: {answer}\n\nIs the candidate answer substantively correct and complete relative to the reference? Answer yes or no."
        try:
            resp = await async_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
                max_completion_tokens=4096, temperature=1.0,
            )
            reply = resp.choices[0].message.content.strip().lower()
        except Exception as exc:
            print(f"  LLM correctness grade failed: {exc}")
            return None
        if reply.startswith("yes"):
            return 1.0
        if reply.startswith("no"):
            return 0.0
        return None

    answers = load_jsonl(ANSWERS_PATH)
    test_set = load_jsonl(Path("data/test_set.jsonl"))
    ts_lookup = {r["query"]: r for r in test_set}

    clauses = load_jsonl(Path("data/clauses.jsonl"))
    clause_lookup = {c["clause_id"]: c["text"] for c in clauses}
    faith_inputs = [
        dict(user_input=r["query"], response=r["answer"],
             retrieved_contexts=[clause_lookup[cid] for cid in r["retrieved"] if cid in clause_lookup])
        for r in answers
    ]
    rel_inputs = [dict(user_input=r["query"], response=r["answer"]) for r in answers]

    print(f"Scoring faithfulness + answer_relevancy for {len(answers)} rows...")
    faith_results = await metric_faithfulness.abatch_score(faith_inputs)
    rel_results = await metric_answer_relevancy.abatch_score(rel_inputs)

    with SCORES_PATH.open("w", encoding="utf-8") as f:
        for ref_name in ["reference_14b", "reference_72b"]:
            ref_short = ref_name.removeprefix("reference_")
            answer_texts = [r["answer"] for r in answers]
            reference_texts = [ts_lookup[r["query"]][ref_name] for r in answers]
            print(f"Scoring correctness against {ref_short}...")
            sims = await cosine_similarities_batch(answer_texts, reference_texts)
            grades = await asyncio.gather(*[
                llm_correctness_grade(r["query"], ref, r["answer"])
                for r, ref in zip(answers, reference_texts)
            ])
            for row, f_res, r_res, sim, grade in zip(answers, faith_results, rel_results, sims, grades):
                out = {
                    "query": row["query"], "query_type": row["query_type"], "config": "hybrid",
                    "variant": "p2_two_step_extraction", "reference_model": ref_short,
                    "faithfulness": f_res.value, "answer_relevancy": r_res.value,
                    "similarity": sim, "llm_grade": grade,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done: scores written to {SCORES_PATH}")


def score() -> None:
    asyncio.run(_score_async())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("extract", "synthesize", "score"):
        print("Usage: uv run python experiment/p2_two_step_extraction.py {extract|synthesize|score}")
        sys.exit(1)
    if sys.argv[1] == "extract":
        extract()
    elif sys.argv[1] == "synthesize":
        synthesize()
    else:
        score()
