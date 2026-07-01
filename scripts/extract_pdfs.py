"""Extract raw text from corpus PDFs and save to data/interim/ as JSON caches.
Run from the implementation/ folder: .venv\\Scripts\\python scripts\\extract_pdfs.py"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import extract_pdf_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("extract_pdfs")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDFS = {
    "mlr_2017": RAW_DIR / "mlr_2017.pdf",
    "jmlsg_1":  RAW_DIR / "jmlsg_part1.pdf",
    "jmlsg_2":  RAW_DIR / "jmlsg_part2.pdf",
    "fca_fcg":  RAW_DIR / "fca_fcg.pdf",
    "fatf_40":  RAW_DIR / "fatf_40.pdf",
}


def extract(name: str, pdf_path: Path) -> None:
    out = OUT_DIR / f"{name}_raw.json"
    if out.exists():
        logger.info("%s: cache exists, skipping", name)
        return
    if not pdf_path.exists():
        logger.warning("%s: not found at %s, skipping", name, pdf_path)
        return
    pages = extract_pdf_pages(pdf_path, source=name)
    out.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("%s: saved %d pages to %s", name, len(pages), out)


for name, path in PDFS.items():
    extract(name, path)

logger.info("Done.")
