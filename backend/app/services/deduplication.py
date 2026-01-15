"""
Event Deduplication Service

Prevents creating duplicate events for ongoing violations.
Only creates new events when:
- A new violation starts (person was compliant, now has missing PPE)
- The violation type changes (different PPE items missing)
- A violation ends (person becomes compliant)

Updates existing events when violations end with end_frame and duration.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ActiveViolation:
    """Tracks an active ongoing violation."""
    event_id: str
    person_id: str
    missing_ppe: Set[str]
    start_frame: int
    start_timestamp: datetime
    last_frame: int
    video_source: str


class DeduplicationManager:
    """
    Manages event deduplication to prevent flooding the database
    with duplicate events for the same ongoing violation.
    """

    def __init__(self):
        # Track active violations: (person_id, video_source) -> ActiveViolation
        self.active_violations: Dict[Tuple[str, str], ActiveViolation] = {}

    def should_create_event(
        self,
        person_id: str,
        video_source: str,
        missing_ppe: List[str],
        frame_number: int,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Determine if a new event should be created.

        Args:
            person_id: ID of the person
            video_source: Video file being processed
            missing_ppe: List of currently missing PPE items
            frame_number: Current frame number

        Returns:
            Tuple of:
                - should_create: Whether to create a new event
                - ended_event_id: ID of event that just ended (if any)
                - reason: Reason for creating/not creating ('new', 'changed', 'ended', None)
        """
        key = (person_id, video_source)
        current_missing = set(missing_ppe)
        active = self.active_violations.get(key)

        # Case 1: No active violation and no current violation
        if not active and not current_missing:
            return False, None, None

        # Case 2: No active violation but new violation detected
        if not active and current_missing:
            return True, None, "new"

        # Case 3: Active violation but person is now compliant
        if active and not current_missing:
            ended_event_id = active.event_id
            del self.active_violations[key]
            return False, ended_event_id, "ended"

        # Case 4: Active violation and still has violations
        if active and current_missing:
            # Check if the missing PPE has changed
            if current_missing != active.missing_ppe:
                # Violation type changed - end old, create new
                ended_event_id = active.event_id
                del self.active_violations[key]
                return True, ended_event_id, "changed"
            else:
                # Same violation continuing - just update last_frame
                active.last_frame = frame_number
                return False, None, "continuing"

        return False, None, None

    def register_event(
        self,
        event_id: str,
        person_id: str,
        video_source: str,
        missing_ppe: List[str],
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        """Register a newly created event as an active violation."""
        key = (person_id, video_source)
        self.active_violations[key] = ActiveViolation(
            event_id=event_id,
            person_id=person_id,
            missing_ppe=set(missing_ppe),
            start_frame=frame_number,
            start_timestamp=timestamp,
            last_frame=frame_number,
            video_source=video_source,
        )

    def get_active_violation(
        self, person_id: str, video_source: str
    ) -> Optional[ActiveViolation]:
        """Get the active violation for a person if any."""
        return self.active_violations.get((person_id, video_source))

    def get_violation_duration(
        self, person_id: str, video_source: str, current_frame: int
    ) -> int:
        """Get the duration in frames for an active violation."""
        active = self.get_active_violation(person_id, video_source)
        if active:
            return current_frame - active.start_frame + 1
        return 0

    def finalize_video(self, video_source: str) -> List[Tuple[str, int]]:
        """
        Finalize all active violations for a video (when processing ends).

        Returns list of (event_id, last_frame) for events that need to be closed.
        """
        to_close = []
        keys_to_remove = []

        for key, violation in self.active_violations.items():
            if violation.video_source == video_source:
                to_close.append((violation.event_id, violation.last_frame))
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.active_violations[key]

        return to_close

    def clear(self) -> None:
        """Clear all tracked violations."""
        self.active_violations.clear()

    def get_stats(self) -> Dict:
        """Get statistics about active violations."""
        return {
            "active_violations": len(self.active_violations),
            "by_video": {
                video: len([v for v in self.active_violations.values() if v.video_source == video])
                for video in set(v.video_source for v in self.active_violations.values())
            },
        }


# Singleton instance
_deduplication_manager: Optional[DeduplicationManager] = None


def get_deduplication_manager() -> DeduplicationManager:
    """Get the singleton deduplication manager instance."""
    global _deduplication_manager
    if _deduplication_manager is None:
        _deduplication_manager = DeduplicationManager()
    return _deduplication_manager
