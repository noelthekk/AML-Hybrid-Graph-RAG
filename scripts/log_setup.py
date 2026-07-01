"""Notebook logging setup: shared across all implementation notebooks.

Every notebook opened with an identical block that created a timestamped log
file under logs/, reset the root handlers (so re-running the cell does not
duplicate output), and streamed to stdout so log lines render as normal cell
output rather than red stderr text. This module collapses that boilerplate
into a single call.

Usage (scripts/ is already on sys.path via each notebook's imports cell):

    from log_setup import setup_logging
    logger = setup_logging("01_corpus_and_retrievers")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "aml_rag"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(notebook, *, level=logging.INFO, log_dir="logs", name=LOGGER_NAME):
    """Configure root logging for a notebook and return its logger.

    notebook : short id used in the log filename, e.g. "01_corpus_and_retrievers".
    level    : root log level (default logging.INFO).
    log_dir  : directory for log files, created if missing (default "logs").
    name     : name of the logger returned (default "aml_rag").

    Writes to logs/<YYYY-MM-DD_HH-MM-SS>_<notebook>.log and to stdout. Existing
    root handlers are closed and replaced (force=True), so the cell is safe to
    re-run without duplicating output.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{notebook}.log"

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(name)
    logger.info("=== Notebook %s started, log: %s ===", notebook, log_file.name)
    return logger
