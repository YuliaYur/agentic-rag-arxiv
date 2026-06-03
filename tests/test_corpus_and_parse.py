"""Tests for corpus metadata loading and the pure PDF-layout logic.

The PyMuPDF call itself isn't tested (it's I/O), but the two pieces of logic it
relies on — heading detection and two-column reading order — are pure functions
and are tested here.
"""

from __future__ import annotations

import json

from agentic_rag.ingest.corpus import clean_title, load_corpus, parse_titles
from agentic_rag.ingest.parse import is_heading, order_blocks

# --------------------------- corpus / titles ---------------------------------


def test_clean_title_strips_author_and_alias():
    assert (
        clean_title("Attention Is All You Need (Transformer) — Vaswani et al.")
        == "Attention Is All You Need"
    )
    assert (
        clean_title("BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al.")
        == "BERT: Pre-training of Deep Bidirectional Transformers"
    )
    assert clean_title("RoBERTa - Liu et al.") == "RoBERTa"


def test_parse_titles_from_sources_table():
    md = (
        "| # | Year | Paper | arXiv | PDF |\n"
        "|---|------|-------|-------|-----|\n"
        "| 1 | 2017 | Attention Is All You Need (Transformer) — Vaswani et al. | "
        "[1706.03762](https://arxiv.org/abs/1706.03762) | [pdf](x) |\n"
        "| 3 | 2018 | BERT: Pre-training — Devlin et al. | "
        "[1810.04805](https://arxiv.org/abs/1810.04805) | [pdf](x) |\n"
    )
    titles = parse_titles(md)
    assert titles["1706.03762"] == "Attention Is All You Need"
    assert titles["1810.04805"] == "BERT: Pre-training"


def test_load_corpus_merges_manifest_and_titles(tmp_path):
    manifest = [
        {
            "arxiv_id": "1706.03762",
            "slug": "transformer",
            "file": "transformer.pdf",
            "status": "downloaded",
        },
        {
            "arxiv_id": "9999.99999",
            "slug": "no_title_paper",
            "file": "x.pdf",
            "status": "downloaded",
        },
    ]
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    spath = tmp_path / "SOURCES.md"
    spath.write_text(
        "| 1 | 2017 | Attention Is All You Need (Transformer) — Vaswani et al. | "
        "[1706.03762](https://arxiv.org/abs/1706.03762) | [pdf](x) |\n",
        encoding="utf-8",
    )
    papers = load_corpus(mpath, spath)
    by_id = {p.arxiv_id: p for p in papers}
    assert by_id["1706.03762"].title == "Attention Is All You Need"
    # Falls back to a de-slugified title when SOURCES.md has no row.
    assert by_id["9999.99999"].title == "No Title Paper"


# --------------------------- heading detection --------------------------------


def test_is_heading_numbered():
    assert is_heading("3 Pre-training", font_size=10, body_size=10)
    assert is_heading("3.1 Model Architecture", font_size=10, body_size=10)


def test_is_heading_unnumbered_keywords():
    assert is_heading("Abstract", font_size=10, body_size=10)
    assert is_heading("References", font_size=10, body_size=10)


def test_is_heading_by_font_size():
    assert is_heading("Scaling Laws", font_size=14, body_size=10)


def test_not_heading_for_body_text():
    body = "We train a deep bidirectional transformer on a large corpus of text."
    assert not is_heading(body, font_size=10, body_size=10)
    # A short line that ends like a normal sentence isn't a heading.
    assert not is_heading("This is fine.", font_size=10, body_size=10)


# --------------------------- two-column reading order -------------------------


def test_order_blocks_two_column_with_full_width_header():
    pw = 600.0  # page width; mid = 300
    # (bbox=(x0,y0,x1,y1), text, font_size)
    blocks = [
        ((10, 0, 590, 30), "TITLE", 16),  # full-width header (top)
        ((40, 100, 280, 140), "left-top", 10),  # left column
        ((40, 200, 280, 240), "left-bottom", 10),
        ((320, 100, 560, 140), "right-top", 10),  # right column
        ((320, 200, 560, 240), "right-bottom", 10),
    ]
    ordered = [t for t, _ in order_blocks(blocks, pw)]
    assert ordered == ["TITLE", "left-top", "left-bottom", "right-top", "right-bottom"]


def test_order_blocks_full_width_separates_bands():
    pw = 600.0
    blocks = [
        ((40, 100, 280, 140), "band1-left", 10),
        ((320, 100, 560, 140), "band1-right", 10),
        ((10, 200, 590, 230), "FULL-WIDTH-FIGURE", 10),  # separator
        ((40, 300, 280, 340), "band2-left", 10),
        ((320, 300, 560, 340), "band2-right", 10),
    ]
    ordered = [t for t, _ in order_blocks(blocks, pw)]
    assert ordered == [
        "band1-left",
        "band1-right",
        "FULL-WIDTH-FIGURE",
        "band2-left",
        "band2-right",
    ]
