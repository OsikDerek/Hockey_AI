"""Analytical verification of quiz logic without a browser.

Loads a positions JSON and computes:
  - Event-type histogram of quiz-eligible events
  - Decision-distribution per event type (what did the actual player do?)
  - Hypothetical scores for naive baselines (always SHOOT, always PASS,
    always pick the modal decision, etc.)

Useful for:
  - Verifying the events JSON has expected shape
  - Catching imbalanced clips ("100% of shot_vs_pass were dumps")
  - Sanity-checking the quiz score widget — if "always SHOOT" gives 70%,
    the quiz is probably too easy on this clip and we should pick a
    more varied test bed.

Usage:
    python scripts/quiz_simulate.py output/<basename>_positions.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


QUIZ_OPTIONS = {
    "shot_vs_pass":       ["shot", "pass", "dump"],
    "odd_man_rush":       ["shot", "pass", "deke"],
    "zone_entry":         ["carry", "dump", "pass_in"],
    "breakout":           ["carry", "rim", "direct_pass", "chip"],
    "missed_opportunity": ["shot", "pass", "hold"],
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: quiz_simulate.py <positions.json>")
        return 1
    payload = json.loads(Path(sys.argv[1]).read_text())
    events = [e for e in payload.get("events", []) if e["event_type"] in QUIZ_OPTIONS]

    print(f"Quiz-eligible events: {len(events)}")
    if not events:
        return 0

    # Per-event-type histogram of actual decisions
    by_type = defaultdict(list)
    for e in events:
        actual = (e.get("decision_made") or "").lower()
        # Normalize missed_X → "hold" since that's how the quiz scores match
        if e["event_type"] == "missed_opportunity":
            actual = "hold"
        by_type[e["event_type"]].append(actual)

    print("\nActual decision distribution per event type:")
    for ev_type, decisions in sorted(by_type.items()):
        c = Counter(decisions)
        total = len(decisions)
        modal = c.most_common(1)[0]
        print(f"  {ev_type} (n={total}):")
        for opt in QUIZ_OPTIONS[ev_type]:
            n = c.get(opt, 0)
            print(f"    {opt:>14}: {n:>3} ({100*n/total:.0f}%)")
        # Anything outside the menu options (would be unscored)
        other = total - sum(c.get(o, 0) for o in QUIZ_OPTIONS[ev_type])
        if other > 0:
            print(f"    {'(off-menu)':>14}: {other:>3}  <- quiz can't score these")
        print(f"    -> modal pick '{modal[0]}' would score {100*modal[1]/total:.0f}%")

    # Naive baselines — what does always-X score across all eligible events?
    print("\nBaseline scores (always pick the same option, summed across all events):")
    baselines = ["shot", "pass", "dump", "carry", "deke", "hold"]
    for pick in baselines:
        matched = 0
        for e in events:
            actual = (e.get("decision_made") or "").lower()
            if e["event_type"] == "missed_opportunity":
                if pick == "hold":
                    matched += 1
            else:
                if pick == actual:
                    matched += 1
        print(f"  always {pick:>5}: {matched}/{len(events)} = {100*matched/len(events):.0f}%")

    # AI rating distribution (good/warning/poor) — how varied is the
    # AI's coaching feedback across these events?
    print("\nAI rating distribution:")
    rcount = Counter(e.get("rating") or "neutral" for e in events)
    for r, n in sorted(rcount.items()):
        print(f"  {r}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
