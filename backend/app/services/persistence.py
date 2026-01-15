from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..services.event_service import EventService
from ..services.person_service import PersonService
from ..services.deduplication import get_deduplication_manager, DeduplicationManager


class PersistenceManager:
    """Coordinates persistence of events and person records with deduplication."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.event_service = EventService(session)
        self.person_service = PersonService(session)
        self.dedup_manager = get_deduplication_manager()

    async def persist_frame_results(
        self, result: Dict[str, Any], snapshot_frame
    ) -> Dict[str, Any]:
        """
        Persist events and update person records from a frame result.

        Uses deduplication to prevent creating duplicate events for ongoing violations.
        Only creates new events when:
        - A new violation starts
        - The violation type changes (different missing PPE)

        Returns:
            Dict with 'created_events' and 'closed_events' counts
        """
        persons = result.get("persons", [])
        frame_number = result.get("frame_number", 0)
        timestamp_str = result.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

        created_events = 0
        closed_events = 0

        # Process each person in the frame
        for person in persons:
            person_id = person.get("person_id")
            if not person_id:
                continue

            # Skip track-only IDs for face-recognized persons check
            if not person_id.startswith("track_"):
                face_embedding = person.get("face_embedding")
                await self.person_service.get_or_create_person(person_id, face_embedding)

            # Get violation info from temporal filter results
            is_stable_violation = person.get("stable_violation", False)
            stable_missing_ppe = person.get("stable_missing_ppe", [])
            video_source = result.get("video_source", "unknown")

            # Use deduplication to determine if we should create an event
            should_create, ended_event_id, reason = self.dedup_manager.should_create_event(
                person_id=person_id,
                video_source=video_source,
                missing_ppe=stable_missing_ppe if is_stable_violation else [],
                frame_number=frame_number,
            )

            # Close ended event if any
            if ended_event_id:
                await self.event_service.close_event(
                    event_id=ended_event_id,
                    end_frame=frame_number - 1,  # Ended on previous frame
                    end_timestamp=timestamp,
                )
                closed_events += 1

            # Create new event if needed
            if should_create and is_stable_violation:
                # Generate event ID
                event_id = str(uuid4())

                # Save snapshot for new violations
                snapshot_path = None
                if settings.ENABLE_SNAPSHOT_CAPTURE and snapshot_frame is not None:
                    filename = f"{person_id}_{frame_number}.jpg"
                    snapshot_path = await self.event_service.save_snapshot(
                        snapshot_frame, settings.SNAPSHOTS_DIR, filename
                    )

                # Create the event
                event = await self.event_service.create_event(
                    person_id=person_id,
                    track_id=person.get("track_id"),
                    timestamp=timestamp,
                    video_source=video_source,
                    frame_number=frame_number,
                    detected_ppe=person.get("detected_ppe", []),
                    missing_ppe=stable_missing_ppe,
                    is_violation=True,
                    detection_confidence=person.get("detection_confidence"),
                    snapshot_path=snapshot_path,
                    start_frame=frame_number,
                )

                # Register with deduplication manager
                self.dedup_manager.register_event(
                    event_id=event.id,
                    person_id=person_id,
                    video_source=video_source,
                    missing_ppe=stable_missing_ppe,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )

                # Update person stats
                if not person_id.startswith("track_"):
                    await self.person_service.increment_event_counts(person_id, True)

                created_events += 1

        await self.session.commit()

        return {
            "created_events": created_events,
            "closed_events": closed_events,
        }

    async def finalize_video_processing(
        self, video_source: str
    ) -> int:
        """
        Finalize all active violations when video processing completes.

        Closes any ongoing events that weren't explicitly ended.
        Returns the number of events closed.
        """
        events_to_close = self.dedup_manager.finalize_video(video_source)

        for event_id, last_frame in events_to_close:
            await self.event_service.close_event(
                event_id=event_id,
                end_frame=last_frame,
                end_timestamp=datetime.now(),
            )

        await self.session.commit()
        return len(events_to_close)
