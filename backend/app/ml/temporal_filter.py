"""
Temporal Filter for Stable Violation Detection

Uses a sliding window buffer to prevent flickering alerts.
Only triggers violations that persist for multiple consecutive frames.
"""

from collections import defaultdict, deque
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ViolationState:
    """State of a violation for temporal filtering."""

    missing_ppe: Set[str]
    frame_count: int
    first_seen: datetime
    last_seen: datetime


class TemporalFilter:
    """
    Filters detection results to ensure temporal consistency.

    Only reports violations that persist for a minimum number of frames,
    reducing false positives from momentary detection failures.
    """

    def __init__(self, buffer_size: int = 3, min_frames_for_violation: int = 3):
        """
        Args:
            buffer_size: Number of frames to keep in history
            min_frames_for_violation: Minimum consecutive frames with violation to trigger
        """
        self.buffer_size = buffer_size
        self.min_frames = min_frames_for_violation

        # Track violations per person: person_id -> deque of missing_ppe sets
        self.violation_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )

        # Track active violations
        self.active_violations: Dict[str, ViolationState] = {}

    def update(self, person_id: str, missing_ppe: List[str]) -> Dict[str, Any]:
        """
        Update the filter with new detection results.

        Args:
            person_id: Unique identifier for the person
            missing_ppe: List of missing PPE items in current frame

        Returns:
            Dict with:
                - is_violation: Whether to trigger a violation alert
                - stable_missing_ppe: PPE that's been missing consistently
                - violation_duration: How many frames the violation has persisted
        """
        missing_set = set(missing_ppe)

        # Add to history
        self.violation_history[person_id].append(missing_set)
        history = self.violation_history[person_id]

        # Need enough history
        if len(history) < self.min_frames:
            return {
                "is_violation": False,
                "stable_missing_ppe": [],
                "violation_duration": 0,
            }

        # Find PPE that's been missing in all recent frames
        stable_missing = set.intersection(*list(history))

        now = datetime.now()

        if stable_missing:
            # Update or create violation state
            if person_id in self.active_violations:
                state = self.active_violations[person_id]
                state.missing_ppe = stable_missing
                state.frame_count += 1
                state.last_seen = now
            else:
                self.active_violations[person_id] = ViolationState(
                    missing_ppe=stable_missing,
                    frame_count=1,
                    first_seen=now,
                    last_seen=now,
                )

            return {
                "is_violation": True,
                "stable_missing_ppe": list(stable_missing),
                "violation_duration": self.active_violations[person_id].frame_count,
            }
        else:
            # Clear violation if it existed
            if person_id in self.active_violations:
                del self.active_violations[person_id]

            return {
                "is_violation": False,
                "stable_missing_ppe": [],
                "violation_duration": 0,
            }

    def get_active_violations(self) -> Dict[str, ViolationState]:
        """Get all currently active violations."""
        return dict(self.active_violations)

    def clear_person(self, person_id: str):
        """Clear history for a specific person."""
        if person_id in self.violation_history:
            del self.violation_history[person_id]
        if person_id in self.active_violations:
            del self.active_violations[person_id]

    def clear_all(self):
        """Clear all history."""
        self.violation_history.clear()
        self.active_violations.clear()


# Singleton instance
_filter = None


def get_temporal_filter() -> TemporalFilter:
    global _filter
    if _filter is None:
        from ..core.config import settings

        _filter = TemporalFilter(
            buffer_size=settings.TEMPORAL_BUFFER_SIZE,
            min_frames_for_violation=settings.TEMPORAL_BUFFER_SIZE,
        )
    return _filter
