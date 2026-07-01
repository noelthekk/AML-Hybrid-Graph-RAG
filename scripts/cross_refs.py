"""Cross-reference extraction from AML regulatory clause text.

Extracts explicit mentions of other clauses within clause text and resolves
them to clause_ids in the corpus. Used to build CROSS_REFERENCES edges in
the knowledge graph.

Supported source patterns:
  mlr_2017  : "Regulation 28", "regulation 28(1)(a)", "Part 3"
  poca_2002 : "section 327", "sections 327 to 329", "s.327A"
  jmlsg_1/2 : "paragraph 5.3.14", "para 5.3"
  fca_fcg   : "FCG 3.2.1", "paragraph 1.1"
  fatf_40   : "Recommendation 10", "R10"
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CrossRef:
    source_id: str
    target_raw: str
    target_id: Optional[str]  # None if clause not found in corpus


# ---------------------------------------------------------------------------
# Compiled patterns (per source)
# ---------------------------------------------------------------------------

_MLR_REG = re.compile(
    r'\b[Rr]egulation[s]?\s+(\d+)(?:\s*\((\d+)\))?',
)
_POCA_SEC_RANGE = re.compile(
    r'\b[Ss]ections?\s+(\d+[A-Z]?)\s+to\s+(\d+[A-Z]?)\b'
)
_POCA_SEC = re.compile(
    r'\b[Ss]ections?\s+(\d+[A-Z]?)\b'
)
_POCA_SEC_SHORT = re.compile(
    r'\bs\.?\s*(\d+[A-Z]?)\b'
)
_JMLSG_PARA = re.compile(
    r'\b(?:paragraph|para)\.?\s+(\d{1,2}\.\d{1,3}(?:\.\d{1,3})?)\b',
    re.IGNORECASE,
)
_FCA_PARA = re.compile(
    r'\bFCG\s+(\d+\.\d+(?:\.\d+)?)\b|\b(?:paragraph|para)\.?\s+(\d+\.\d+)\b',
    re.IGNORECASE,
)
_FATF_REC = re.compile(
    r'\b[Rr]ecommendation\s+(\d{1,2})\b|\bR(\d{1,2})\b'
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cross_refs(clause: dict, clause_index: dict) -> list[CrossRef]:
    """Return cross-references found in clause text, resolved against clause_index.

    clause       : one record from clauses.jsonl
    clause_index : {clause_id: record} for the whole corpus
    """
    source = clause["source"]
    text = clause["text"]
    cid = clause["clause_id"]
    refs: list[CrossRef] = []

    if source == "mlr_2017":
        refs.extend(_extract_mlr(cid, text, clause_index))

    elif source == "poca_2002":
        refs.extend(_extract_poca(cid, text, clause_index))

    elif source in ("jmlsg_1", "jmlsg_2"):
        refs.extend(_extract_jmlsg(cid, source, text, clause_index))

    elif source == "fca_fcg":
        refs.extend(_extract_fca(cid, text, clause_index))

    elif source == "fatf_40":
        refs.extend(_extract_fatf(cid, text, clause_index))

    # Deduplicate by (source_id, target_id)
    seen: set[tuple] = set()
    deduped: list[CrossRef] = []
    for r in refs:
        key = (r.source_id, r.target_id or r.target_raw)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def refs_to_records(refs: list[CrossRef]) -> list[dict]:
    return [
        {
            "source_id": r.source_id,
            "target_raw": r.target_raw,
            "target_id": r.target_id,
        }
        for r in refs
    ]


# ---------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------

def _resolve(target_id: str, clause_index: dict) -> Optional[str]:
    return target_id if target_id in clause_index else None


def _extract_mlr(cid: str, text: str, idx: dict) -> list[CrossRef]:
    refs = []
    for m in _MLR_REG.finditer(text):
        reg_num = int(m.group(1))
        tid = f"mlr_2017_reg_{reg_num}"
        if tid != cid:
            refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                  target_id=_resolve(tid, idx)))
    return refs


def _extract_poca(cid: str, text: str, idx: dict) -> list[CrossRef]:
    refs = []
    # Ranges: "sections 327 to 329"
    for m in _POCA_SEC_RANGE.finditer(text):
        s1, s2 = m.group(1), m.group(2)
        # Expand numeric ranges only (skip if alpha suffix)
        try:
            n1, n2 = int(s1), int(s2)
            for n in range(n1, n2 + 1):
                tid = f"poca_2002_s{n}"
                if tid != cid:
                    refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                          target_id=_resolve(tid, idx)))
        except ValueError:
            tid = f"poca_2002_s{s1}"
            if tid != cid:
                refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                      target_id=_resolve(tid, idx)))

    # Individual sections: "section 327A"
    range_spans = {(m.start(), m.end()) for m in _POCA_SEC_RANGE.finditer(text)}
    for m in _POCA_SEC.finditer(text):
        if any(rs <= m.start() and m.end() <= re_ for rs, re_ in range_spans):
            continue  # already handled by range
        sec = m.group(1)
        tid = f"poca_2002_s{sec}"
        if tid != cid:
            refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                  target_id=_resolve(tid, idx)))

    return refs


def _extract_jmlsg(cid: str, source: str, text: str, idx: dict) -> list[CrossRef]:
    refs = []
    for m in _JMLSG_PARA.finditer(text):
        para = m.group(1)
        # Try same part first, then the other
        for src in (source, "jmlsg_1" if source == "jmlsg_2" else "jmlsg_2"):
            tid = f"{src}_{para}"
            if tid != cid and tid in idx:
                refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                      target_id=tid))
                break
        else:
            tid = f"{source}_{para}"
            if tid != cid:
                refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                      target_id=None))
    return refs


def _extract_fca(cid: str, text: str, idx: dict) -> list[CrossRef]:
    refs = []
    for m in _FCA_PARA.finditer(text):
        para = m.group(1) or m.group(2)
        if para:
            tid = f"fca_fcg_{para}"
            if tid != cid:
                refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                      target_id=_resolve(tid, idx)))
    return refs


def _extract_fatf(cid: str, text: str, idx: dict) -> list[CrossRef]:
    refs = []
    for m in _FATF_REC.finditer(text):
        num = m.group(1) or m.group(2)
        if num:
            tid = f"fatf_40_R{num}"
            if tid != cid:
                refs.append(CrossRef(source_id=cid, target_raw=m.group(0),
                                      target_id=_resolve(tid, idx)))
    return refs
