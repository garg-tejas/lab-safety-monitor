from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import settings
from .core.database import init_db, async_session
from .services.person_service import PersonService
from .ml.pipeline import get_pipeline
from .api.routes import (
    events_router,
    persons_router,
    stats_router,
    stream_router,
    ws_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME}...")
    await init_db()
    print("Database initialized")

    # Ensure directories exist
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline and load persisted person embeddings
    pipeline = get_pipeline()
    pipeline.initialize()
    async with async_session() as session:
        person_service = PersonService(session)
        embeddings = await person_service.get_all_embeddings()
        if embeddings:
            pipeline.load_known_persons(embeddings)
            print(f"Loaded {len(embeddings)} known persons")

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered laboratory safety compliance monitoring system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(events_router, prefix="/api")
app.include_router(persons_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(stream_router, prefix="/api")
app.include_router(ws_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"name": settings.APP_NAME, "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
