"""Decision detector registry for game analysis.

Each decision type is a separate detector class that identifies
when a tactical decision occurs and what the player chose to do.
"""

from .shot_vs_pass import ShotVsPassDetector
from .zone_entry import ZoneEntryDetector
from .breakout import BreakoutDetector

DECISION_REGISTRY = {
    "shot_vs_pass": ShotVsPassDetector,
    "zone_entry": ZoneEntryDetector,
    "breakout": BreakoutDetector,
}
