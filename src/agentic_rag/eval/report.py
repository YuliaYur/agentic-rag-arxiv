"""Render an eval report dict as a Markdown comparison table.

The headline is the per-system aggregate table (baseline vs agent across all
metrics), followed by a compact per-question breakdown so you can see *where* a
system wins or loses.
"""

from __future__ import annotations

from .runner import METRIC_KEYS

_LABELS = {
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer rel.",
    "context_precision": "Ctx prec.",
    "context_recall": "Ctx recall",
    "judge_overall_norm": "Judge",
}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def render_markdown(report: dict, title: str = "Evaluation results") -> str:
    systems = report["systems"]
    agg = report["aggregates"]
    cfg = report["config"]
    lines: list[str] = [f"# {title}", ""]
    lines.append(
        f"Golden questions: **{cfg['n_items']}** · k=**{cfg['k']}** · "
        f"RAGAS-style metrics: {'on' if cfg['use_ragas'] else 'off'} · "
        f"LLM-judge: {'on' if cfg['use_judge'] else 'off'}. "
        "All scores in [0,1], higher is better."
    )
    lines.append("")

    # --- aggregate table: metrics x systems ---
    header = "| Metric | " + " | ".join(systems) + " | Winner |"
    sep = "|" + "---|" * (len(systems) + 2)
    lines += [header, sep]
    for key in METRIC_KEYS:
        row_vals = [agg[s].get(key) for s in systems]
        cells = " | ".join(_fmt(v) for v in row_vals)
        nums = [
            (s, v) for s, v in zip(systems, row_vals, strict=True) if isinstance(v, (int, float))
        ]
        winner = "—"
        if nums:
            best = max(nums, key=lambda t: t[1])
            ties = [s for s, v in nums if abs(v - best[1]) < 1e-9]
            winner = "tie" if len(ties) > 1 else best[0]
        lines.append(f"| {_LABELS[key]} | {cells} | {winner} |")
    errs = " · ".join(f"{s}: {agg[s].get('n_errors', 0)} errors" for s in systems)
    lines += ["", f"_Errors — {errs}._", ""]

    # --- per-question judge + recall breakdown ---
    lines += ["## Per-question (judge overall, normalized)", ""]
    head = "| id | type | " + " | ".join(systems) + " |"
    lines += [head, "|" + "---|" * (len(systems) + 2)]
    for q in report["per_question"]:
        cells = []
        for s in systems:
            sysd = q["systems"].get(s, {})
            if "error" in sysd:
                cells.append("ERR")
            else:
                cells.append(_fmt(sysd.get("scores", {}).get("judge_overall_norm")))
        lines.append(f"| {q['id']} | {q['type']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
