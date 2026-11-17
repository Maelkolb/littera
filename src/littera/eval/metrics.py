from __future__ import annotations

import difflib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from littera.eval.ink import InkProfile
from littera.types import Placement, word_spans


def visible_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def letter_width(row: dict, profile: InkProfile | None = None) -> float:
    count = max(1, visible_count(row["text"]))
    gold_words = row.get("gold_words") or []
    if gold_words:
        span = max(w["x1"] for w in gold_words) - min(w["x0"] for w in gold_words)
    elif profile is not None:
        x0, x1 = profile.extent()
        span = x1 - x0
    else:
        span = float(row["width"])
    return max(1.0, span / count)


def predicted_cut(placement: Placement, i: int) -> float:
    chars, text = placement.chars, placement.text
    if not chars:
        return 0.0
    if i <= 0:
        return chars[0].x0
    if i >= len(chars):
        return chars[-1].x1
    if text[i].isspace():
        return chars[i - 1].x1
    if text[i - 1].isspace():
        return chars[i].x0
    return (chars[i - 1].x1 + chars[i].x0) / 2


def predicted_word(placement: Placement, i0: int, i1: int) -> tuple[float, float, float, float]:
    for word in placement.words:
        if word.i0 == i0 and word.i1 == i1:
            return word.x0, word.x1, word.q, word.fit
    chars = placement.chars[i0:i1]
    if not chars:
        return 0.0, 0.0, 0.0, 0.0
    return chars[0].x0, chars[-1].x1, min(c.q for c in chars), placement.line_fit


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def changed_words(text: str, true_text: str | None) -> list[bool]:
    words = [text[a:b] for a, b in word_spans(text)]
    if true_text is None or text == true_text:
        return [False] * len(words)
    true_words = [true_text[a:b] for a, b in word_spans(true_text)]
    changed = [True] * len(words)
    for tag, a0, a1, _, _ in difflib.SequenceMatcher(a=words, b=true_words, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(a0, a1):
                changed[k] = False
    return changed


def row_scores(placement: Placement, row: dict, profile: InkProfile | None = None) -> dict:
    text = row["text"]
    width = letter_width(row, profile)
    scores: dict = {"uid": row["uid"], "source": row.get("source", ""), "kind": row.get("kind", ""), "letter_width": width,
                    "words": [], "cuts": [], "gaps": [], "line_fit": placement.line_fit}
    for gold in row.get("gold_words") or []:
        x0, x1, q, fit = predicted_word(placement, gold["i0"], gold["i1"])
        scores["words"].append({"e0": abs(x0 - gold["x0"]) / width, "e1": abs(x1 - gold["x1"]) / width,
                                "iou": interval_iou(x0, x1, gold["x0"], gold["x1"]), "q": q, "fit": fit})
    chars = placement.chars
    for gold in row.get("gold_cuts") or []:
        i = gold["i"]
        if not (0 < i < len(text)) or text[i].isspace() or text[i - 1].isspace() or len(chars) != len(text):
            continue
        scores["cuts"].append({"e": abs(predicted_cut(placement, i) - gold["x"]) / width, "q": min(chars[i - 1].q, chars[i].q)})
    gold_at = {gold["i"]: gold["x"] for gold in row.get("gold_cuts") or []}
    if len(chars) == len(text):
        for i, ch in enumerate(text):
            if ch.isspace() or i not in gold_at or i + 1 not in gold_at:
                continue
            left = abs(chars[i].x0 - gold_at[i]) / width
            right = abs(chars[i].x1 - gold_at[i + 1]) / width
            scores.setdefault("chars", []).append({"e": max(left, right), "q": chars[i].q})
    if profile is not None:
        for left, right in zip(placement.words, placement.words[1:]):
            scores["gaps"].append(profile.is_gap((left.x1 + right.x0) / 2))
    if row.get("label") is not None:
        scores["label"] = int(row["label"])
        scores["variant"] = row.get("variant") or ("right" if row["label"] == 1 else "wrong")
        changed = changed_words(text, row.get("true_text"))
        fits = {(w.i0, w.i1): w.fit for w in placement.words}
        scores["word_fits"] = [(fits.get(span, placement.line_fit), 0 if bad else 1) for span, bad in zip(word_spans(text), changed)]
    return scores


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float | None:
    if not confidences:
        return None
    conf = np.clip(np.asarray(confidences, dtype=np.float64), 0, 1)
    hits = np.asarray(correct, dtype=np.float64)
    index = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.sum() * abs(hits[members].mean() - conf[members].mean())
    return float(total / len(conf))


def summarise_errors(errors: Sequence[float]) -> dict:
    if not errors:
        return {"n": 0}
    values = np.asarray(errors, dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "within_0.25": float((values <= 0.25).mean()),
            "within_0.5": float((values <= 0.5).mean())}


def rounded(value):
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def aggregate(all_scores: Iterable[dict], by_source: bool = True) -> dict:
    all_scores = list(all_scores)
    result = aggregate_group(all_scores)
    if by_source:
        groups = defaultdict(list)
        for scores in all_scores:
            groups[scores["source"]].append(scores)
        if len(groups) > 1:
            result["by_source"] = {source: aggregate_group(group) for source, group in sorted(groups.items())}
    return rounded(result)


def aggregate_group(all_scores: list[dict]) -> dict:
    words = [w for s in all_scores for w in s["words"]]
    cuts = [c for s in all_scores for c in s["cuts"]]
    gaps = [g for s in all_scores for g in s["gaps"]]
    result: dict = {"lines": len(all_scores)}
    if words:
        edges = [w["e0"] for w in words] + [w["e1"] for w in words]
        word_right = [max(w["e0"], w["e1"]) <= 0.5 for w in words]
        word_q_mean = float(np.mean([w["q"] for w in words]))
        result["word"] = {**summarise_errors(edges), "words": len(words), "iou": float(np.mean([w["iou"] for w in words])),
                          "both_ends_within_0.5": float(np.mean(word_right)),
                          "q_auc": auc([w["q"] for w in words], word_right),
                          "q_ece": expected_calibration_error([w["q"] for w in words], word_right), "q_mean": word_q_mean}
    if cuts:
        cut_right = [c["e"] <= 0.5 for c in cuts]
        result["letter"] = {**summarise_errors([c["e"] for c in cuts]), "q_auc": auc([c["q"] for c in cuts], cut_right),
                            "q_ece": expected_calibration_error([c["q"] for c in cuts], cut_right)}
    chars = [c for s in all_scores for c in s.get("chars", [])]
    if chars:
        char_right = [c["e"] <= 0.25 for c in chars]
        result["char"] = {"n": len(chars), "both_cuts_within_0.25": float(np.mean(char_right)),
                          "q_auc": auc([c["q"] for c in chars], char_right),
                          "q_ece": expected_calibration_error([c["q"] for c in chars], char_right),
                          "q_mean": float(np.mean([c["q"] for c in chars]))}
    if gaps:
        result["gap_rate"] = float(np.mean(gaps))
        result["gaps"] = len(gaps)
    labelled = [s for s in all_scores if "label" in s]
    if labelled:
        fit = {"rows": len(labelled), "line_fit_auc": auc([s["line_fit"] for s in labelled], [s["label"] for s in labelled])}
        right = [s for s in labelled if s["label"] == 1]
        for variant in sorted({s["variant"] for s in labelled if s["label"] == 0}):
            wrong = [s for s in labelled if s.get("variant") == variant and s["label"] == 0]
            pair = right + wrong
            fit[f"line_fit_auc_{variant}"] = auc([s["line_fit"] for s in pair], [s["label"] for s in pair])
        word_pairs = [p for s in labelled for p in s.get("word_fits", [])]
        if word_pairs:
            fit["word_fit_auc"] = auc([p[0] for p in word_pairs], [p[1] for p in word_pairs])
        fit["line_fit_mean_right"] = float(np.mean([s["line_fit"] for s in right])) if right else None
        result["fit"] = fit
    return result
