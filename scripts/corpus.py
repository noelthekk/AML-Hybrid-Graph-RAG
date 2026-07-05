"""Corpus assembly: PDF download, text extraction, and clause segmentation."""
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import pdfplumber
from liteparse import LiteParse

logger = logging.getLogger(__name__)

AUTO_DOWNLOAD = {
    "mlr_2017":  "https://www.legislation.gov.uk/uksi/2017/692/pdfs/uksi_20170692_en.pdf",
    "poca_2002": "https://www.legislation.gov.uk/ukpga/2002/29/pdfs/ukpga_20020029_en.pdf",
    "fatf_40":   "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Recommendations%202012.pdf",
}
MANUAL_PDFS = {
    "jmlsg_1": "jmlsg_part1.pdf",
    "jmlsg_2": "jmlsg_part2.pdf",
    "fca_fcg": "fca_fcg.pdf",
}
POCA_XML_URL = "https://www.legislation.gov.uk/ukpga/2002/29/data.xml"

_HEADERS = {"User-Agent": "Mozilla/5.0 (research/dissertation)"}
# Both regexes below accept a slightly wider pattern than pdfplumber's output strictly
# needs, to also work against LiteParse's output (see extract_pdf_pages_liteparse):
# - LiteParse's OCR fallback sometimes renders the source's em-dash ("28.-(1)") as a
#   plain hyphen; verified this widened match is still 218/218-identical against
#   pdfplumber's own output for mlr_2017 (2026-07-05).
# - The real schedule headings are one line, "SCHEDULE 1 - Professional Bodies" (title
#   included) - pdfplumber's reconstruction happens to split the title onto its own
#   line, which is why requiring nothing else on the line worked before, coincidentally.
#   Kept case-sensitive (no re.IGNORECASE): body-text mentions like "Schedule 5" use
#   mixed case, so this still doesn't false-match those.
_MLR_CLAUSE_RE = re.compile(r"(?m)^(\d{1,3})\.([‐-―\-]|[ \t])")
_MLR_SCHEDULE_RE = re.compile(r"(?m)^SCHEDULE\s+(\d+)\b")
_JMLSG_PARA_RE = re.compile(r"(?m)^(\d{1,2}\.\d{1,3}[A-Za-z]?(?:\.\d{1,3}[A-Za-z]?)?)\s")
_FATF_HEADER_RE = re.compile(
    r"THE FATF RECOMMENDATIONS\s*\nINTERNATIONAL STANDARDS ON COMBATING MONEY LAUNDERING[^\n]*\n",
    re.MULTILINE,
)
_FATF_REC_RE = re.compile(r"(?m)^(\d{1,2})\.\s+[A-Z]")
_FATF_INR_RE = re.compile(r"(?m)^INTERPRETIVE NOTE TO RECOMMENDATION\s+(\d{1,2})")


def download_pdf(name: str, url: str, dest: Path) -> bool:
    if dest.exists():
        logger.info("SKIP %s: already exists (%d KB)", name, dest.stat().st_size // 1024)
        return True
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        if "application/pdf" not in resp.headers.get("Content-Type", ""):
            logger.warning("WARN %s: unexpected content type", name)
            return False
        dest.write_bytes(resp.content)
        logger.info("OK   %s: %d KB", name, len(resp.content) // 1024)
        return True
    except Exception as e:
        logger.error("FAIL %s: %s", name, e)
        return False


def ensure_corpus(raw_dir: Path) -> bool:
    """Download auto-downloadable PDFs and warn about manual ones. Returns True if all present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in AUTO_DOWNLOAD.items():
        download_pdf(name, url, raw_dir / f"{name}.pdf")
    all_present = True
    for name, fname in MANUAL_PDFS.items():
        p = raw_dir / fname
        if p.exists():
            logger.info("OK   %s (%d KB)", name, p.stat().st_size // 1024)
        else:
            logger.warning("MISSING %s: download from jmlsg.org.uk / handbook.fca.org.uk", name)
            all_present = False
    return all_present


def extract_pdf_pages(pdf_path: Path, source: str = "") -> list[dict]:
    """Extract text page by page from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i % 20 == 0:
                logger.info("%s: page %d / %d", source or pdf_path.name, i + 1, total)
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page_num": i + 1, "text": text.strip()})
    logger.info("%s: %d pages extracted", source or pdf_path.name, len(pages))
    return pages


def extract_pdf_pages_liteparse(pdf_path: Path, source: str = "") -> list[dict]:
    """Extract text page by page from a PDF using LiteParse. Same output shape as
    extract_pdf_pages (pdfplumber), so both are interchangeable with the segmentation
    functions below.

    Validated 2026-07-05 against the real corpus: exact match on mlr_2017 (218/218),
    jmlsg_2 (901/901), fca_fcg (49/49); 561/567 (98.9%) on jmlsg_1 (a handful of
    paragraphs have a marginal citation note landing before rather than after the
    paragraph number, breaking the line-start match - accepted as a known minor gap).

    Not used for fatf_40: LiteParse's reading-order reconstruction breaks down on that
    document's styled headings (colored/bold recommendation titles interleaved with
    body text), producing genuinely garbled text no amount of segmentation-regex tuning
    can fix. pdfplumber extracts fatf_40 correctly - see extract_pdfs.py, which uses
    pdfplumber for that source specifically. Full investigation, including the page
    render and positional evidence: 01_corpus_and_retrievers.md's 2026-07-05 Change log
    entry, and figures/fatf_40_page15_liteparse_diagnosis.png.
    """
    parser = LiteParse()
    result = parser.parse(str(pdf_path))
    pages = []
    for page in result.pages:
        lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in page.text.split("\n")]
        text = "\n".join(lines).strip()
        if text:
            pages.append({"page_num": page.page_num, "text": text})
    logger.info("%s: %d pages extracted (liteparse)", source or pdf_path.name, len(pages))
    return pages


def load_json_cache(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_toc_page(text: str, threshold: float = 0.4) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    return sum(1 for line in lines if re.search(r"\s+\d{1,3}$", line)) / len(lines) >= threshold


def _filter_toc(pages: list[dict], source: str = "") -> list[dict]:
    body = [p for p in pages if not _is_toc_page(p["text"])]
    logger.info("%s: TOC filter, kept %d / %d pages", source, len(body), len(pages))
    return body


def _clause(clause_id: str, source: str, marker: str, text: str) -> dict:
    return {"clause_id": clause_id, "source": source, "marker": marker, "text": text, "cross_refs": []}


def segment_mlr(pages: list[dict], source: str = "mlr_2017") -> list[dict]:
    pages = _filter_toc(pages, source)
    full_text = "\n".join(p["text"] for p in pages)

    sch_match = _MLR_SCHEDULE_RE.search(full_text)
    main_text = full_text[: sch_match.start()] if sch_match else full_text
    schedule_text = full_text[sch_match.start() :] if sch_match else ""

    clauses = []
    matches = list(_MLR_CLAUSE_RE.finditer(main_text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(main_text)
        text = main_text[m.start() : end].strip()
        if len(text) >= 50:
            clauses.append(_clause(f"{source}_reg_{m.group(1)}", source, f"Regulation {m.group(1)}", text))

    if schedule_text:
        sch_headers = list(_MLR_SCHEDULE_RE.finditer(schedule_text))
        for j, sch in enumerate(sch_headers):
            sch_end = sch_headers[j + 1].start() if j + 1 < len(sch_headers) else len(schedule_text)
            sch_body = schedule_text[sch.end() : sch_end]
            para_matches = list(_MLR_CLAUSE_RE.finditer(sch_body))
            for k, pm in enumerate(para_matches):
                end = para_matches[k + 1].start() if k + 1 < len(para_matches) else len(sch_body)
                text = sch_body[pm.start() : end].strip()
                if len(text) >= 50:
                    clauses.append(_clause(
                        f"{source}_sch{sch.group(1)}_para{pm.group(1)}",
                        source,
                        f"Schedule {sch.group(1)} Para {pm.group(1)}",
                        text,
                    ))

    logger.info("%s: %d clauses", source, len(clauses))
    return clauses


def fetch_poca_xml(cache_path: Path) -> ET.Element:
    if cache_path.exists():
        logger.info("Loading POCA XML from cache")
        content = cache_path.read_bytes()
    else:
        logger.info("Downloading POCA XML from legislation.gov.uk")
        resp = requests.get(POCA_XML_URL, timeout=120)
        resp.raise_for_status()
        content = resp.content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        logger.info("Saved to %s", cache_path)
    return ET.fromstring(content)


def segment_poca(xml_root: ET.Element, source: str = "poca_2002") -> list[dict]:
    def _tag(el):
        return el.tag.split("}")[-1]

    def _get_text(el):
        parts = [el.text.strip()] if el.text and el.text.strip() else []
        for child in el:
            t = _get_text(child)
            if t:
                parts.append(t)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        return " ".join(parts)

    clauses = []
    for p1 in xml_root.iter():
        if _tag(p1) != "P1":
            continue
        sec_id = p1.get("id", "")
        if not sec_id.startswith("section-"):
            continue
        text = _get_text(p1)
        if any(c.isalpha() for c in text):
            sec_num = sec_id[len("section-"):]
            clauses.append(_clause(f"{source}_s{sec_num}", source, f"Section {sec_num}", text))

    logger.info("%s: %d clauses", source, len(clauses))
    return clauses


def segment_jmlsg(pages: list[dict], source: str) -> list[dict]:
    """Segment JMLSG Parts I/II and FCA FCG: all use the same paragraph numbering."""
    body = _filter_toc(pages, source)
    full_text = "\n".join(p["text"] for p in body)
    matches = list(_JMLSG_PARA_RE.finditer(full_text))

    seen: dict[str, dict] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[m.start() : end].strip()
        if len(text) < 30:
            continue
        cid = f"{source}_{m.group(1)}"
        if cid not in seen or len(text) > len(seen[cid]["text"]):
            seen[cid] = _clause(cid, source, f"Para {m.group(1)}", text)

    clauses = list(seen.values())
    logger.info("%s: %d clauses", source, len(clauses))
    return clauses


def segment_fatf(pages: list[dict], source: str = "fatf_40") -> list[dict]:
    body = _filter_toc(pages, source)
    full_text = _FATF_HEADER_RE.sub("", "\n".join(p["text"] for p in body))

    inr_pos = full_text.find("INTERPRETIVE NOTE TO RECOMMENDATION")
    rec_section = full_text[: inr_pos if inr_pos != -1 else len(full_text)]
    rec_matches = list(_FATF_REC_RE.finditer(rec_section))

    seen: dict[str, dict] = {}
    for i, m in enumerate(rec_matches):
        end = rec_matches[i + 1].start() if i + 1 < len(rec_matches) else len(rec_section)
        text = full_text[m.start() : end].strip()
        if len(text) < 30:
            continue
        cid = f"{source}_R{m.group(1)}"
        if cid not in seen or len(text) > len(seen[cid]["text"]):
            seen[cid] = _clause(cid, source, f"Recommendation {m.group(1)}", text)

    clauses = list(seen.values())
    inr_matches = list(_FATF_INR_RE.finditer(full_text))
    for i, m in enumerate(inr_matches):
        end = inr_matches[i + 1].start() if i + 1 < len(inr_matches) else len(full_text)
        text = full_text[m.start() : end].strip()
        if len(text) >= 30:
            clauses.append(_clause(
                f"{source}_INR{m.group(1)}",
                source,
                f"Interpretive Note to Recommendation {m.group(1)}",
                text,
            ))

    logger.info("%s: %d clauses", source, len(clauses))
    return clauses
