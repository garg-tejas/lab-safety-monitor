"""
Video upload and processing endpoints for recorded video analysis.
Includes annotated video generation for frontend display.
"""

import asyncio
import cv2
import base64
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
from pydantic import BaseModel

from ...ml.pipeline import get_pipeline


class ProcessVideoRequest(BaseModel):
    video_path: str


from ...core.config import settings
from ...core.database import async_session
from ...services.persistence import PersistenceManager


router = APIRouter(prefix="/stream", tags=["video"])


class VideoProcessingManager:
    """Manages video processing jobs."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, video_path: str, filename: str) -> str:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "video_path": video_path,
            "filename": filename,
            "status": "pending",
            "progress": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "violations_count": 0,
            "persons_count": 0,
            "unique_events": 0,  # Deduplicated event count
            "error": None,
            "output_video": None,  # Path to annotated video
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return list(self.jobs.values())


processing_manager = VideoProcessingManager()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported: mp4, avi, mov, mkv, webm",
        )

    # Ensure videos directory exists
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to videos directory
    video_path = settings.VIDEOS_DIR / file.filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return JSONResponse(
        {
            "message": "Video uploaded successfully",
            "filename": file.filename,
            "path": str(video_path),
        }
    )


@router.post("/process")
async def start_video_processing(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start processing an uploaded video.

    Returns a job_id that can be used to track progress.
    """
    video_path = request.video_path

    # Validate video exists
    path = Path(video_path)
    if not path.exists():
        # Try in videos directory
        path = settings.VIDEOS_DIR / video_path
        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"Video not found: {video_path}"
            )

    # Create processing job
    job_id = processing_manager.create_job(str(path), path.name)

    # Start background processing
    background_tasks.add_task(process_video_task, job_id, str(path))

    return {
        "message": "Processing started",
        "job_id": job_id,
        "video_path": str(path),
    }


async def process_video_task(job_id: str, video_path: str):
    """
    Background task to process a video file.

    Creates an annotated output video and uses event deduplication.
    """
    processing_manager.update_job(job_id, status="processing")

    pipeline = get_pipeline()
    pipeline.initialize()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        processing_manager.update_job(
            job_id, status="failed", error=f"Could not open video: {video_path}"
        )
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, int(video_fps / settings.FRAME_SAMPLE_RATE))

    processing_manager.update_job(job_id, total_frames=total_frames)

    cap.release()

    # Setup output video writer for annotated frames
    output_dir = settings.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = Path(video_path).stem
    output_filename = f"{input_filename}_annotated_{job_id[:8]}.mp4"
    output_path = output_dir / output_filename

    # Use mp4v codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = min(settings.FRAME_SAMPLE_RATE, video_fps)
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, output_fps, (frame_width, frame_height)
    )

    processed_count = 0
    total_unique_events = 0
    unique_persons = set()

    try:
        for result in pipeline.process_video(video_path):
            processed_count += 1

            # Add video_source to result for persistence
            result["video_source"] = video_path

            # Write annotated frame to output video
            annotated_frame = result.get("annotated_frame")
            if annotated_frame is not None and video_writer.isOpened():
                video_writer.write(annotated_frame)

            # Persist with deduplication - process all persons, not just events
            async with async_session() as session:
                persistence = PersistenceManager(session)
                persist_result = await persistence.persist_frame_results(
                    result, result.get("annotated_frame")
                )
                total_unique_events += persist_result.get("created_events", 0)

            # Track unique persons
            for person in result.get("persons", []):
                if person.get("person_id"):
                    unique_persons.add(person["person_id"])

            # Update progress
            progress = int((processed_count * frame_skip / total_frames) * 100)
            processing_manager.update_job(
                job_id,
                processed_frames=processed_count,
                progress=min(progress, 100),
                violations_count=total_unique_events,  # Now shows unique events
                unique_events=total_unique_events,
                persons_count=len(unique_persons),
            )

            # Small delay to prevent blocking
            await asyncio.sleep(0.001)

        # Finalize: close any remaining open violations
        async with async_session() as session:
            persistence = PersistenceManager(session)
            closed_count = await persistence.finalize_video_processing(video_path)
            total_unique_events += closed_count

        # Close video writer
        video_writer.release()

        processing_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            violations_count=total_unique_events,
            unique_events=total_unique_events,
            output_video=str(output_path) if output_path.exists() else None,
        )

    except Exception as e:
        video_writer.release()
        processing_manager.update_job(job_id, status="failed", error=str(e))


@router.get("/jobs")
async def list_processing_jobs():
    """List all processing jobs."""
    return {"jobs": processing_manager.list_jobs()}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific processing job."""
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.get("/videos")
async def list_uploaded_videos():
    """List all uploaded videos."""
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        for path in settings.VIDEOS_DIR.glob(ext):
            videos.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
            )

    return {"videos": videos}


@router.delete("/videos/{filename}")
async def delete_video(filename: str):
    """Delete an uploaded video."""
    video_path = settings.VIDEOS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")

    video_path.unlink()
    return {"message": f"Deleted {filename}"}


@router.get("/status")
async def get_processing_status():
    """Get overall processing status."""
    active_jobs = [
        j for j in processing_manager.list_jobs() if j["status"] == "processing"
    ]
    return {
        "active_jobs": len(active_jobs),
        "total_jobs": len(processing_manager.jobs),
    }


@router.get("/processed/{job_id}")
async def get_processed_video(job_id: str):
    """
    Get the annotated/processed video for a completed job.

    Returns the video file with bounding boxes and detection overlays.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_video = job.get("output_video")
    if not output_video or not Path(output_video).exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found. It may have been deleted."
        )

    return FileResponse(
        output_video,
        media_type="video/mp4",
        filename=Path(output_video).name
    )


@router.get("/processed/{job_id}/stream")
async def stream_processed_video(job_id: str):
    """
    Stream the annotated video as MJPEG for real-time display.

    This is useful for displaying the processed video frame-by-frame in the browser.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_video = job.get("output_video")
    if not output_video or not Path(output_video).exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found."
        )

    async def generate_frames():
        """Generate MJPEG frames from the processed video."""
        cap = cv2.VideoCapture(output_video)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = jpeg.tobytes()

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

                # Control playback speed (roughly 10 FPS)
                await asyncio.sleep(0.1)
        finally:
            cap.release()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/processed/list")
async def list_processed_videos():
    """List all processed/annotated videos."""
    processed_dir = settings.PROCESSED_DIR
    if not processed_dir.exists():
        return {"videos": []}

    videos = []
    for path in processed_dir.glob("*.mp4"):
        videos.append({
            "filename": path.name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        })

    return {"videos": videos}


@router.delete("/processed/{job_id}")
async def delete_processed_video(job_id: str):
    """Delete a processed video."""
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    output_video = job.get("output_video")
    if output_video and Path(output_video).exists():
        Path(output_video).unlink()
        processing_manager.update_job(job_id, output_video=None)
        return {"message": f"Deleted processed video for job {job_id}"}

    return {"message": "No processed video to delete"}
