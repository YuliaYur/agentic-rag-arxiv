"""Load + validate the golden evaluation set (``eval/golden_set.jsonl``).

One JSON object per line. The dataset is the heart of the eval story, so loading
is strict: a malformed or incomplete row raises rather than being silently skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingest.config import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.jsonl"

_TYPES = {"factual", "comparative", "multi-hop"}


@dataclass(frozen=True)
class GoldenItem:
    """One evaluation question with its reference answer and expected sources."""

    id: str
    question: str
    type: str  # factual | comparative | multi-hop
    expected_arxiv_ids: list[str]  # the paper(s) that should be retrieved/cited
    reference_answer: str  # ground-truth answer (for recall + judge + correctness)
    notes: str = ""
    status: str = "draft"  # draft (awaiting curation) | seed | reviewed

    @property
    def is_multihop(self) -> bool:
        return self.type in {"comparative", "multi-hop"} or len(self.expected_arxiv_ids) > 1


def load_golden_set(path: Path | str = GOLDEN_PATH) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
        item = _validate(raw, where=f"{path}:{n}")
        if item.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"{path}: golden set is empty")
    return items


def _validate(raw: dict, where: str) -> GoldenItem:
    required = ("id", "question", "type", "expected_arxiv_ids", "reference_answer")
    for key in required:
        if key not in raw or raw[key] in (None, "", []):
            raise ValueError(f"{where}: missing/empty required field {key!r}")
    if raw["type"] not in _TYPES:
        raise ValueError(f"{where}: type must be one of {sorted(_TYPES)}, got {raw['type']!r}")
    if not isinstance(raw["expected_arxiv_ids"], list):
        raise ValueError(f"{where}: expected_arxiv_ids must be a list")
    return GoldenItem(
        id=raw["id"],
        question=raw["question"],
        type=raw["type"],
        expected_arxiv_ids=list(raw["expected_arxiv_ids"]),
        reference_answer=raw["reference_answer"],
        notes=raw.get("notes", ""),
        status=raw.get("status", "draft"),
    )
