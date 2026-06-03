"""Load corpus metadata from local files only (no network).

Two local sources are merged into one record per paper:

* ``data/raw/manifest.json`` -> arxiv_id, slug, pdf filename (written by
  ``scripts/fetch_corpus.py``).
* ``SOURCES.md`` -> the human-readable paper title (the markdown table rows).

Titles come from SOURCES.md because the PDF's own metadata title is frequently
missing or wrong for arXiv papers. The arxiv_id is the join key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Matches a SOURCES.md table row and captures the title cell + the arxiv id from
# its abstract link, e.g.
#   | 1 | 2017 | Attention Is All You Need (Transformer) — Vaswani et al. | [1706.03762](...) | [pdf](...) |
_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*\d{4}\s*\|\s*(?P<title>.+?)\s*\|\s*\[(?P<arxiv_id>\d{4}\.\d{4,5})\]"
)


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    slug: str
    file: str
    title: str

    def pdf_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.file


def clean_title(raw: str) -> str:
    """Strip the trailing author attribution and any parenthetical alias.

    "Attention Is All You Need (Transformer) — Vaswani et al." -> "Attention Is All You Need"
    Handles both the em dash (—) and a plain hyphen used as the author separator.
    """
    # Drop the author tail after an em dash or " - " separator.
    title = re.split(r"\s+[—–-]\s+", raw, maxsplit=1)[0]
    # Drop a trailing "(Alias)" like "(Transformer)" / "(BERT)".
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    return title.strip()


def parse_titles(sources_md: str) -> dict[str, str]:
    """Parse SOURCES.md text into {arxiv_id: clean_title}."""
    titles: dict[str, str] = {}
    for line in sources_md.splitlines():
        m = _ROW_RE.match(line.strip())
        if m:
            titles[m.group("arxiv_id")] = clean_title(m.group("title"))
    return titles


def load_corpus(manifest_path: Path, sources_md: Path) -> list[Paper]:
    """Merge manifest + titles into a list of Paper records."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    titles = parse_titles(Path(sources_md).read_text(encoding="utf-8"))

    papers: list[Paper] = []
    for entry in manifest:
        arxiv_id = entry["arxiv_id"]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                slug=entry["slug"],
                file=entry["file"],
                # Fall back to a de-slugified name if the title is somehow absent.
                title=titles.get(arxiv_id, entry["slug"].replace("_", " ").title()),
            )
        )
    return papers
