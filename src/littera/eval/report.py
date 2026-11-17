from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cell(value, digits: int = 2, percent: bool = False) -> str:
    if value is None:
        return "-"
    if percent:
        return f"{100 * value:.1f}"
    return f"{value:.{digits}f}"


def placement_table(scores: dict, sets: list[str], placers: list[str]) -> str:
    lines = ["| set | placer | lines | words | word edge error median (letters) | within 0.25 / 0.5 (%) | word x-IoU | "
             "letter cut error median | letter within 0.25 / 0.5 (%) | ink-gap rate (%) | word q AUC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for set_name in sets:
        for placer in placers:
            result = scores.get(f"{placer}|{set_name}")
            if not result:
                continue
            word, letter = result.get("word", {}), result.get("letter", {})
            lines.append(
                f"| {set_name} | {placer} | {result.get('lines', 0)} | {word.get('words', '-')} | {cell(word.get('median'))} | "
                f"{cell(word.get('within_0.25'), percent=True)} / {cell(word.get('within_0.5'), percent=True)} | {cell(word.get('iou'))} | "
                f"{cell(letter.get('median'))} | {cell(letter.get('within_0.25'), percent=True)} / "
                f"{cell(letter.get('within_0.5'), percent=True)} | "
                f"{cell(result.get('gap_rate'), percent=True)} | {cell(word.get('q_auc'))} |")
    return "\n".join(lines)


def fit_table(scores: dict, set_name: str, placers: list[str]) -> str:
    variants = ["swap", "swap-doc", "replace", "delete"]
    header = " | ".join(f"AUC {v}" for v in variants)
    lines = [f"| placer | rows | line_fit AUC | {header} | word fit AUC (replaced word) | mean line_fit, right text |",
             "|---|---|---|" + "---|" * len(variants) + "---|---|"]
    for placer in placers:
        fit = (scores.get(f"{placer}|{set_name}") or {}).get("fit")
        if not fit:
            continue
        lines.append(f"| {placer} | {fit['rows']} | {cell(fit.get('line_fit_auc'), 3)} | "
                     + " | ".join(cell(fit.get(f"line_fit_auc_{v}"), 3) for v in variants)
                     + f" | {cell(fit.get('word_fit_auc'), 3)} | {cell(fit.get('line_fit_mean_right'), 3)} |")
    return "\n".join(lines)


def by_source_table(scores: dict, set_name: str, placers: list[str], metric: str = "word") -> str:
    sources = sorted({s for p in placers for s in ((scores.get(f"{p}|{set_name}") or {}).get("by_source") or {})})
    if not sources:
        return ""
    lines = ["| source | " + " | ".join(placers) + " |", "|---|" + "---|" * len(placers)]
    for source in sources:
        cells = []
        for placer in placers:
            group = ((scores.get(f"{placer}|{set_name}") or {}).get("by_source") or {}).get(source, {})
            if metric == "gap":
                cells.append(cell(group.get("gap_rate"), percent=True))
            elif metric == "fit":
                cells.append(cell((group.get("fit") or {}).get("line_fit_auc"), 3))
            else:
                cells.append(cell((group.get(metric) or {}).get("within_0.5"), percent=True))
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littera.eval.report")
    parser.add_argument("--scores", default="runs/eval/baselines.json")
    parser.add_argument("--placers", default="even,kraken-k2,kraken-mccatmus")
    parser.add_argument("--sets", default="gw20-words,iam-words,synth-heldfonts,mismatch-heldout,letter-gold-candidates")
    args = parser.parse_args(argv)
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    placers, sets = args.placers.split(","), args.sets.split(",")
    print(placement_table(scores, sets, placers))
    print()
    print(fit_table(scores, "mismatch-heldout", placers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
