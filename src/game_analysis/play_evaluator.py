"""Play evaluator for game decision analysis.

Scores detected decisions and suggests better alternatives using the
knowledge base. Supports team play style presets that bias evaluation
toward the team's preferred approach.

Play styles influence how decisions are rated:
- "possession": Favors controlled plays, passing, carrying the puck
- "physical": Favors aggressive plays, dumping and chasing, body contact
- "speed": Favors fast transitions, quick releases, stretch passes
- "balanced": No bias, purely outcome-based evaluation
"""

import os
import yaml
from typing import Optional

from .game_context import GameEvent


# Built-in play style definitions
PLAY_STYLES = {
    "balanced": {
        "display_name": "Balanced",
        "description": "No style bias — evaluate purely on outcome",
        "biases": {},
    },
    "possession": {
        "display_name": "Possession Hockey",
        "description": "Favor keeping the puck, controlled plays, cycling",
        "biases": {
            "shot_vs_pass": {"pass": 0.15, "shot": 0.0, "dump": -0.2},
            "zone_entry": {"carry": 0.2, "pass_in": 0.1, "dump": -0.3},
            "breakout": {"carry": 0.15, "direct_pass": 0.1, "rim": -0.1, "chip": -0.2},
            "odd_man_rush": {"pass": 0.1, "shot": 0.0, "deke": -0.1},
            "forecheck": {"moderate": 0.1, "aggressive": -0.05, "passive": -0.1},
            "defensive_play": {"contain": 0.1, "gap_close": 0.05, "pinch": -0.15},
        },
    },
    "physical": {
        "display_name": "Physical / Dump & Chase",
        "description": "Favor aggressive plays, dump-ins, forechecking",
        "biases": {
            "shot_vs_pass": {"shot": 0.15, "dump": 0.1, "pass": -0.1},
            "zone_entry": {"dump": 0.2, "carry": 0.0, "pass_in": -0.1},
            "breakout": {"rim": 0.15, "chip": 0.1, "carry": -0.1, "direct_pass": -0.05},
            "odd_man_rush": {"shot": 0.1, "deke": 0.05, "pass": -0.05},
            "forecheck": {"aggressive": 0.2, "moderate": 0.0, "passive": -0.2},
            "defensive_play": {"gap_close": 0.15, "pinch": 0.1, "contain": -0.1},
        },
    },
    "speed": {
        "display_name": "Speed / Transition",
        "description": "Favor fast plays, quick releases, stretch passes",
        "biases": {
            "shot_vs_pass": {"shot": 0.1, "pass": 0.1, "dump": -0.15},
            "zone_entry": {"carry": 0.15, "pass_in": 0.15, "dump": -0.1},
            "breakout": {"direct_pass": 0.2, "carry": 0.1, "rim": -0.1, "chip": -0.15},
            "odd_man_rush": {"shot": 0.1, "pass": 0.1, "deke": -0.1},
            "forecheck": {"aggressive": 0.1, "moderate": 0.05, "passive": -0.15},
            "defensive_play": {"gap_close": 0.1, "contain": 0.0, "pinch": -0.1},
        },
    },
    "defensive": {
        "display_name": "Defensive / Low Risk",
        "description": "Favor safe plays, avoid turnovers, chip when in doubt",
        "biases": {
            "shot_vs_pass": {"shot": 0.0, "dump": 0.1, "pass": -0.1},
            "zone_entry": {"dump": 0.15, "carry": -0.1, "pass_in": -0.05},
            "breakout": {"chip": 0.2, "rim": 0.15, "carry": -0.15, "direct_pass": -0.1},
            "odd_man_rush": {"shot": 0.1, "pass": -0.05, "deke": -0.15},
            "forecheck": {"passive": 0.15, "moderate": 0.05, "aggressive": -0.15},
            "defensive_play": {"contain": 0.15, "gap_close": 0.1, "pinch": -0.2},
        },
    },
}


class PlayEvaluator:
    """Evaluate game decisions and suggest alternatives.

    Loads evaluation criteria from knowledge_base/game_situations/ YAML
    files and applies team play style biases.

    Usage:
        evaluator = PlayEvaluator(play_style="possession")
        evaluation = evaluator.evaluate(event)
    """

    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base/game_situations",
        play_style: str = "balanced",
    ):
        """
        Args:
            knowledge_base_dir: Path to game situation YAML files.
            play_style: Team play style preset. One of: balanced, possession,
                physical, speed, defensive.
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.situations = {}

        if play_style not in PLAY_STYLES:
            print(f"  Warning: Unknown play style '{play_style}', using 'balanced'")
            play_style = "balanced"

        self.play_style = PLAY_STYLES[play_style]
        self.play_style_name = play_style
        print(f"  Play style: {self.play_style['display_name']}")

        # Load YAML files
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load all game situation YAML files."""
        if not os.path.isdir(self.knowledge_base_dir):
            print(f"  No knowledge base at {self.knowledge_base_dir}")
            return

        for fname in os.listdir(self.knowledge_base_dir):
            if fname.endswith(".yaml") and not fname.startswith("_"):
                path = os.path.join(self.knowledge_base_dir, fname)
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    name = fname.replace(".yaml", "")
                    self.situations[name] = data.get("situation", data)
                    print(f"  Loaded situation: {name}")
                except Exception as e:
                    print(f"  Warning: Failed to load {fname}: {e}")

    def evaluate(self, event: GameEvent) -> dict:
        """Evaluate a single game event.

        Args:
            event: GameEvent with decision_made and context.

        Returns:
            Evaluation dict with rating, reasoning, alternative, style_note.
        """
        evaluation = {
            "rating": "neutral",          # good, warning, poor, neutral
            "score": 0.5,                 # 0.0 (terrible) to 1.0 (excellent)
            "reasoning": "",
            "alternative": None,          # Suggested better play
            "style_note": None,           # Play style context
            "context_factors": {},        # What influenced the rating
        }

        # Get situation config from knowledge base
        situation = self.situations.get(event.event_type)

        if situation:
            evaluation = self._evaluate_with_yaml(event, situation, evaluation)
        else:
            evaluation = self._evaluate_heuristic(event, evaluation)

        # Apply play style bias
        evaluation = self._apply_style_bias(event, evaluation)

        # Classify rating from score
        score = evaluation["score"]
        if score >= 0.7:
            evaluation["rating"] = "good"
        elif score >= 0.4:
            evaluation["rating"] = "warning"
        else:
            evaluation["rating"] = "poor"

        return evaluation

    def _evaluate_with_yaml(
        self, event: GameEvent, situation: dict, evaluation: dict
    ) -> dict:
        """Evaluate using YAML-defined criteria."""
        checks = situation.get("checks", {})
        feedback = situation.get("feedback", {})

        score_total = 0.0
        score_count = 0
        reasons = []

        for check_name, check_cfg in checks.items():
            metric_name = check_cfg.get("metric", check_name)
            metric_value = event.context.get(metric_name)

            if metric_value is None:
                continue

            good_range = check_cfg.get("good_range", [])
            warning_range = check_cfg.get("warning_range", [])

            if len(good_range) == 2 and good_range[0] <= metric_value <= good_range[1]:
                score_total += 1.0
                if "good" in check_cfg.get("feedback", {}):
                    reasons.append(check_cfg["feedback"]["good"])
            elif len(warning_range) == 2 and warning_range[0] <= metric_value <= warning_range[1]:
                score_total += 0.5
                if "warning" in check_cfg.get("feedback", {}):
                    reasons.append(check_cfg["feedback"]["warning"])
            else:
                score_total += 0.0
                if "poor" in check_cfg.get("feedback", {}):
                    reasons.append(check_cfg["feedback"]["poor"])

            score_count += 1
            evaluation["context_factors"][check_name] = metric_value

        if score_count > 0:
            evaluation["score"] = score_total / score_count

        # Build reasoning
        decision = event.decision_made
        if decision in feedback:
            reasons.insert(0, feedback[decision])

        evaluation["reasoning"] = " | ".join(reasons) if reasons else f"Decision: {decision}"

        # Suggest alternative from situation config
        alternatives = situation.get("alternatives", {})
        if decision in alternatives:
            evaluation["alternative"] = alternatives[decision]

        return evaluation

    def _evaluate_heuristic(self, event: GameEvent, evaluation: dict) -> dict:
        """Fallback heuristic evaluation when no YAML exists."""
        ctx = event.context
        decision = event.decision_made

        if event.event_type == "shot_vs_pass":
            blockers = ctx.get("defenders_blocking", 0)
            open_mates = ctx.get("open_teammates", 0)

            if decision == "shot":
                if blockers == 0:
                    evaluation["score"] = 0.9
                    evaluation["reasoning"] = "Open lane — good shot decision"
                elif blockers <= 1 and open_mates == 0:
                    evaluation["score"] = 0.7
                    evaluation["reasoning"] = "Partial screen but no better option"
                elif open_mates >= 2:
                    evaluation["score"] = 0.3
                    evaluation["reasoning"] = f"Shot through {blockers} defenders with {open_mates} open teammates"
                    evaluation["alternative"] = "Pass to open teammate for better angle"
                else:
                    evaluation["score"] = 0.5
                    evaluation["reasoning"] = "Contested shot"

            elif decision == "pass":
                evaluation["score"] = 0.6
                evaluation["reasoning"] = "Pass — depending on completion"

            elif decision == "dump":
                evaluation["score"] = 0.3
                evaluation["reasoning"] = "Puck dumped — lost possession opportunity"

        elif event.event_type == "zone_entry":
            if decision == "carry":
                evaluation["score"] = 0.75
                evaluation["reasoning"] = "Carried puck into zone — maintained possession"
            elif decision == "dump":
                evaluation["score"] = 0.35
                evaluation["reasoning"] = "Dumped puck in — need to win puck battle"
                evaluation["alternative"] = "Look for carry or pass-in to maintain possession"
            elif decision == "pass_in":
                evaluation["score"] = 0.7
                evaluation["reasoning"] = "Passed puck into zone"

        elif event.event_type == "breakout":
            success = ctx.get("success", False)
            dzone_time = ctx.get("dzone_time", 0)

            if success:
                evaluation["score"] = 0.8
                evaluation["reasoning"] = f"Clean breakout after {dzone_time}s in D-zone"
            else:
                evaluation["score"] = 0.3
                evaluation["reasoning"] = f"Breakout failed — spent {dzone_time}s in D-zone"

            if decision == "carry" and not success:
                evaluation["alternative"] = "Consider quick pass or rim to relieve pressure"
            elif decision == "chip" and success:
                evaluation["reasoning"] += " (chip worked but risky — consider controlled options)"

        elif event.event_type == "odd_man_rush":
            rush_type = ctx.get("rush_type", "unknown")
            defenders = ctx.get("defenders_back", 0)

            if decision == "shot":
                # Game theory: shoot 75-82% of the time
                evaluation["score"] = 0.75
                evaluation["reasoning"] = f"{rush_type} — shot is correct 75-82% of the time per game theory"
            elif decision == "pass":
                # Pass creates highest xG when defender plays shot, but risky
                evaluation["score"] = 0.6
                evaluation["reasoning"] = f"{rush_type} — pass creates highest xG (0.214) when defender plays shot"
                if defenders <= 1:
                    evaluation["score"] = 0.7
                    evaluation["reasoning"] += " — good read with only one defender back"
            elif decision == "deke":
                evaluation["score"] = 0.35
                evaluation["reasoning"] = f"{rush_type} — deke is low-percentage on odd-man rushes"
                evaluation["alternative"] = "Shoot or pass — dekes waste the numbers advantage"

        elif event.event_type == "forecheck":
            pressure = ctx.get("pressure_distance", 0)

            if decision == "aggressive":
                evaluation["score"] = 0.7
                evaluation["reasoning"] = "Aggressive forecheck — creating pressure"
                if ctx.get("num_players_pressuring", 0) >= 3:
                    evaluation["score"] = 0.5
                    evaluation["reasoning"] = "All forwards deep — no safety valve, risk of odd-man rush"
                    evaluation["alternative"] = "Keep F3 high as safety — never all three forwards deep"
            elif decision == "moderate":
                evaluation["score"] = 0.65
                evaluation["reasoning"] = "Balanced forecheck — pressure with coverage"
            elif decision == "passive":
                evaluation["score"] = 0.5
                evaluation["reasoning"] = "Passive forecheck — giving opponent free exits"
                evaluation["alternative"] = "Consider more pressure unless protecting a lead"

        elif event.event_type == "defensive_play":
            gap = ctx.get("gap_distance", 0)

            if decision == "gap_close":
                evaluation["score"] = 0.75
                evaluation["reasoning"] = "Closed the gap — forcing the attacker to make a decision"
            elif decision == "contain":
                evaluation["score"] = 0.6
                evaluation["reasoning"] = "Contained the play — waiting for support"
                if gap > 2.0:
                    evaluation["score"] = 0.4
                    evaluation["reasoning"] = "Gap too wide — attacker has too much time and space"
                    evaluation["alternative"] = "Close the gap to 4-6 feet — do not back up to the net"
            elif decision == "pinch":
                evaluation["score"] = 0.45
                evaluation["reasoning"] = "Pinched aggressively — high-risk play"
                evaluation["alternative"] = "Pinching risks odd-man rush if you do not win the puck"

        return evaluation

    def _apply_style_bias(self, event: GameEvent, evaluation: dict) -> dict:
        """Apply team play style bias to the evaluation score."""
        biases = self.play_style.get("biases", {})
        event_biases = biases.get(event.event_type, {})

        decision = event.decision_made
        if decision in event_biases:
            bias = event_biases[decision]
            old_score = evaluation["score"]
            evaluation["score"] = max(0.0, min(1.0, old_score + bias))

            if abs(bias) > 0.05:
                style_name = self.play_style["display_name"]
                if bias > 0:
                    evaluation["style_note"] = (
                        f"Aligns with {style_name} approach (+{bias:.0%} style bonus)"
                    )
                else:
                    evaluation["style_note"] = (
                        f"Against {style_name} philosophy ({bias:.0%} style penalty)"
                    )

        return evaluation

    def evaluate_all(self, events: list) -> list:
        """Evaluate all events and attach evaluation dicts."""
        for event in events:
            event.evaluation = self.evaluate(event)
        return events
