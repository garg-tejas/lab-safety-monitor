import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_logging() -> None:
    """
    Set up logging for the application.

    Strategy:
    - Root logger: WARNING (keeps third-party noise quiet)
    - app.*        : INFO  (our own code is fully visible)
    - sqlalchemy.engine: WARNING — suppresses the per-query flood that
      comes from echo=True in DEBUG mode; real DB errors still surface
    - uvicorn.*    : leave at their own defaults
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    # Avoid duplicate handlers if uvicorn already added one
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Our application code — fully visible
    logging.getLogger("app").setLevel(logging.INFO)

    # Silence SQLAlchemy per-query echo (errors/warnings still come through)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    # Keep uvicorn access / error logs at their natural level
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


# Configure logging before any application imports so that loggers
# created during module-level initialisation pick up the right level.
configure_logging()

from .api.routes import (
    events_router,
    persons_router,
    stats_router,
    stream_router,
    ws_router,
)
from .core.config import settings
from .core.database import async_session, init_db
from .ml.pipeline import get_pipeline
from .services.person_service import PersonService


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
    try:
        pipeline.initialize()
    except RuntimeError as exc:
        # Keep diagnostics/API available while inference is explicitly unavailable.
        logging.getLogger(__name__).error("Detection pipeline unavailable: %s", exc)
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
# In debug mode, allow common development origins
# In production, use specific origins from settings
if settings.DEBUG:
    # Allow common development ports and origins
    debug_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ] + settings.CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=debug_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
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
    inference = get_pipeline().status()
    return {"status": "healthy" if inference["ready"] else "degraded", "inference": inference}
