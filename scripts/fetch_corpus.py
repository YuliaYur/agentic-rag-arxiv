"""Reproducible corpus fetcher for the agentic RAG project.

Downloads the 20 transformer-lineage papers listed in SOURCES.md from arXiv
into data/raw/. PDFs are intentionally NOT committed to the repo; this script
is the reproducible ingestion entry point instead.

Usage:
    python scripts/fetch_corpus.py                 # download all, skip existing
    python scripts/fetch_corpus.py --out data/raw  # custom output dir
    python scripts/fetch_corpus.py --force         # re-download even if present

Be a polite arXiv client: this script throttles requests and sets a
descriptive User-Agent. See SOURCES.md -> Licensing for details.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

# (arxiv_id, slug) -- slug becomes the filename stem. See SOURCES.md.
PAPERS: list[tuple[str, str]] = [
    ("1706.03762", "transformer_attention_is_all_you_need"),
    ("1802.05365", "elmo_deep_contextualized_word_representations"),
    ("1810.04805", "bert"),
    ("1907.11692", "roberta"),
    ("1910.10683", "t5_transfer_learning_limits"),
    ("2003.10555", "electra"),
    ("2001.08361", "scaling_laws_for_neural_lm"),
    ("2005.14165", "gpt3_few_shot_learners"),
    ("2203.15556", "chinchilla_compute_optimal"),
    ("2004.05150", "longformer"),
    ("2205.14135", "flashattention"),
    ("2106.09685", "lora"),
    ("1908.10084", "sentence_bert"),
    ("2004.04906", "dpr_dense_passage_retrieval"),
    ("2005.11401", "rag_retrieval_augmented_generation"),
    ("2010.11929", "vit_image_worth_16x16_words"),
    ("2012.12877", "deit_data_efficient_image_transformers"),
    ("2103.14030", "swin_transformer"),
    ("2103.00020", "clip"),
    ("2111.06377", "mae_masked_autoencoders"),
]

PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
# Identify the client and give a contact point, per arXiv's automated-access guidance.
USER_AGENT = "agentic-rag-portfolio/1.0 (corpus fetch for personal research; +https://github.com/yourname/yourrepo)"
DELAY_SECONDS = 3.0  # polite throttle between downloads


def download(arxiv_id: str, dest: Path) -> None:
    url = PDF_URL.format(arxiv_id=arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{arxiv_id}: response is not a PDF (got {len(data)} bytes)")
    dest.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    failures: list[str] = []

    for i, (arxiv_id, slug) in enumerate(PAPERS, start=1):
        dest = out_dir / f"{slug}.pdf"
        prefix = f"[{i:2d}/{len(PAPERS)}] {arxiv_id}"

        if dest.exists() and not args.force:
            print(f"{prefix}  skip (exists): {dest.name}")
            manifest.append(
                {"arxiv_id": arxiv_id, "slug": slug, "file": dest.name, "status": "cached"}
            )
            continue

        try:
            print(f"{prefix}  downloading -> {dest.name}")
            download(arxiv_id, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"{prefix}  done ({size_kb} KB)")
            manifest.append(
                {"arxiv_id": arxiv_id, "slug": slug, "file": dest.name, "status": "downloaded"}
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{prefix}  FAILED: {exc}")
            failures.append(arxiv_id)
            manifest.append(
                {"arxiv_id": arxiv_id, "slug": slug, "file": dest.name, "status": f"failed: {exc}"}
            )

        # Throttle only between live downloads, not after cached skips.
        if i < len(PAPERS):
            time.sleep(DELAY_SECONDS)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}")
        print(
            "Tip: a failing ID may need a version suffix (e.g. 1706.03762v7). "
            "Check the paper's arXiv abstract page."
        )
        raise SystemExit(1)

    print(f"\nAll {len(PAPERS)} papers present in {out_dir}/")


if __name__ == "__main__":
    main()
