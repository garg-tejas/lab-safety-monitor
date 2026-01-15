from .events import router as events_router
from .persons import router as persons_router
from .stats import router as stats_router
from .stream import router as stream_router

__all__ = ["events_router", "persons_router", "stats_router", "stream_router"]
