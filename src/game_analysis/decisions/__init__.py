"""Decision detector registry for game analysis.

Each decision type is a separate detector class that identifies
when a tactical decision occurs and what the player chose to do.
"""

from .shot_vs_pass import ShotVsPassDetector
from .zone_entry import ZoneEntryDetector
from .breakout import BreakoutDetector
from .odd_man_rush import OddManRushDetector
from .forecheck import ForecheckDetector
from .defensive_play import DefensivePlayDetector
from .missed_opportunity import MissedOpportunityDetector

DECISION_REGISTRY = {
    "shot_vs_pass": ShotVsPassDetector,
    "zone_entry": ZoneEntryDetector,
    "breakout": BreakoutDetector,
    "odd_man_rush": OddManRushDetector,
    "forecheck": ForecheckDetector,
    "defensive_play": DefensivePlayDetector,
    "missed_opportunity": MissedOpportunityDetector,
}
