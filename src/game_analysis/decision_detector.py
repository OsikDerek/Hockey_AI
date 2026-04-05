"""Base protocol for game decision detectors.

Each decision detector identifies when a tactical decision occurs
and classifies what the player chose to do.
"""

from typing import Protocol, runtime_checkable
from .game_context import FrameContext, GameEvent


@runtime_checkable
class BaseDecisionDetector(Protocol):
    """Protocol that all decision detectors must implement."""

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame of game context."""
        ...

    def analyze(self, fps: float) -> list:
        """Analyze all collected frames and return detected events.

        Returns:
            List of GameEvent instances.
        """
        ...
