"""
Run this after setting up the environment to confirm everything is installed and working.
Usage: python verify_env.py
"""
import importlib
import sys

checks = [
    ("pdfplumber", "pdfplumber"),
    ("rank_bm25", "rank-bm25"),
    ("sentence_transformers", "sentence-transformers"),
    ("chromadb", "chromadb"),
    ("langchain_core", "langchain-core"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("ipykernel", "ipykernel"),
    ("jupyter_core", "jupyter"),
]

failed = []
for module, pkg in checks:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "ok")
        print(f"  OK  {pkg} ({version})")
    except ImportError:
        print(f"  MISSING  {pkg}")
        failed.append(pkg)

print()

from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
v = emb.embed_query("test")
print(f"  OK  all-MiniLM-L6-v2 (dim={len(v)})")

print()
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
else:
    print("All checks passed.")
