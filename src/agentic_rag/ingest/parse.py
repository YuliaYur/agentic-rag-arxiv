"""Layout-aware PDF parsing with PyMuPDF (fitz).

Why PyMuPDF (full rationale + the Unstructured / LlamaParse comparison is in
DECISIONS.md): it is fast, pure-local, free, and exposes per-span font sizes and
block bounding boxes. Those two signals are exactly what we need for our
two-column academic PDFs:

* bounding boxes -> reconstruct correct **reading order** (left column fully,
  then right column, with full-width blocks like titles/abstracts/wide tables
  acting as band separators).
* font sizes + numbered-heading regex -> detect **section headers** so chunking
  can stay inside section boundaries and tag each chunk with its section.

Output is a flat, reading-ordered list of ``Block``s, each tagged with its page
(1-based) and the section it belongs to.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import fitz  # PyMuPDF

# A numbered section heading: "1 Introduction", "3.2 Pre-training", "A.1 ...".
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+|[A-Z])(\.\d+)*\.?\s+[A-Z][\w&/-]")
# Unnumbered headings common in arXiv papers.
_UNNUMBERED_HEADINGS = {
    "abstract", "introduction", "related work", "background", "conclusion",
    "conclusions", "references", "acknowledgments", "acknowledgements",
    "appendix", "discussion", "methods", "method", "experiments", "results",
}
# Heading detection: a heading line is short.
_MAX_HEADING_WORDS = 12

FRONTMATTER = "Frontmatter"
REFERENCES = "References"


@dataclass
class Block:
    """One reading-ordered text block from the PDF."""

    text: str
    page: int          # 1-based page number
    section: str       # section title this block belongs to
    is_heading: bool


def _block_font_size(block: dict) -> float:
    """Largest span size in a fitz text block (headings are set in a bigger font)."""
    sizes = [span["size"] for line in block.get("lines", []) for span in line.get("spans", [])]
    return max(sizes) if sizes else 0.0


def _block_text(block: dict) -> str:
    """Join spans/lines of a fitz block into a single normalized string."""
    lines = []
    for line in block.get("lines", []):
        text = "".join(span["text"] for span in line.get("spans", []))
        if text.strip():
            lines.append(text.strip())
    return " ".join(lines).strip()


def _body_font_size(doc: fitz.Document) -> float:
    """Median span size across the document, weighted by character count.

    This is the body-text size; headings are detected relative to it.
    """
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    n = len(span["text"].strip())
                    if n:
                        sizes.extend([round(span["size"], 1)] * n)
    return statistics.median(sizes) if sizes else 10.0


def heading_kind(text: str, font_size: float, body_size: float) -> str | None:
    """Classify a block as a heading and say *how* it was detected.

    Returns "numbered", "keyword", "font", or None. The kind matters to the
    caller: a font-only heading on page 1 is almost always the paper *title*,
    not a section, so it should not open a section. Pure logic, unit-tested.
    """
    stripped = text.strip()
    if not stripped or len(stripped.split()) > _MAX_HEADING_WORDS:
        return None
    if _NUMBERED_HEADING_RE.match(stripped):
        return "numbered"
    if stripped.lower().rstrip(":") in _UNNUMBERED_HEADINGS:
        return "keyword"
    # A noticeably larger, short line that looks title-like.
    if font_size >= body_size + 1.0 and stripped[0].isupper() and not stripped.endswith("."):
        return "font"
    return None


def is_heading(text: str, font_size: float, body_size: float) -> bool:
    """Whether a block is a section heading (any kind)."""
    return heading_kind(text, font_size, body_size) is not None


def order_blocks(blocks: list[tuple[tuple[float, float, float, float], str, float]],
                 page_width: float) -> list[tuple[str, float]]:
    """Reconstruct reading order for a (possibly two-column) page.

    ``blocks`` is a list of (bbox, text, font_size); bbox is (x0, y0, x1, y1).
    Returns (text, font_size) in reading order.

    Algorithm: walk blocks top-to-bottom. Full-width blocks (wider than 55% of
    the page) are band separators emitted in place. Between separators, buffered
    blocks are split into left/right columns by horizontal center and emitted
    left-column-first, each column top-to-bottom.
    """
    mid_x = page_width / 2.0
    full_width_threshold = 0.55 * page_width

    ordered: list[tuple[str, float]] = []
    buffer: list[tuple[tuple[float, float, float, float], str, float]] = []

    def flush() -> None:
        if not buffer:
            return
        left = sorted((b for b in buffer if (b[0][0] + b[0][2]) / 2 < mid_x), key=lambda b: b[0][1])
        right = sorted((b for b in buffer if (b[0][0] + b[0][2]) / 2 >= mid_x), key=lambda b: b[0][1])
        for bbox, text, size in (*left, *right):
            ordered.append((text, size))
        buffer.clear()

    for bbox, text, size in sorted(blocks, key=lambda b: b[0][1]):  # sort by y0
        width = bbox[2] - bbox[0]
        if width >= full_width_threshold:
            flush()
            ordered.append((text, size))
        else:
            buffer.append((bbox, text, size))
    flush()
    return ordered


def parse_pdf(path, *, drop_references: bool = True) -> list[Block]:
    """Parse a PDF into reading-ordered, section-tagged blocks.

    The PDF/network is out of scope for unit tests; the testable logic
    (``is_heading``, ``order_blocks``, chunking) is factored out above and in
    chunk.py. This function wires them to PyMuPDF.
    """
    doc = fitz.open(path)
    try:
        body_size = _body_font_size(doc)
        out: list[Block] = []
        current_section = FRONTMATTER
        hit_references = False
        in_body = False  # flips once we pass the title/abstract into real sections

        for page_index, page in enumerate(doc, start=1):
            raw_blocks = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type", 0) != 0:  # skip images
                    continue
                text = _block_text(block)
                if text:
                    raw_blocks.append((tuple(block["bbox"]), text, _block_font_size(block)))

            for text, size in order_blocks(raw_blocks, page.rect.width):
                kind = heading_kind(text, size, body_size)
                # A font-only "heading" before any real section is the paper
                # title (or similar frontmatter) -> don't open a section for it.
                is_section_heading = kind in ("numbered", "keyword") or (kind == "font" and in_body)
                if is_section_heading:
                    in_body = True
                    current_section = text.strip()
                    if current_section.lower().rstrip(":") == "references":
                        hit_references = True
                        current_section = REFERENCES
                if drop_references and hit_references:
                    continue
                out.append(Block(text=text, page=page_index, section=current_section,
                                 is_heading=is_section_heading))
        return out
    finally:
        doc.close()
