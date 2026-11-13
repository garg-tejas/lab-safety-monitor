"""Explicit failure boundary between unavailable inference and simulated demos."""


class ModelUnavailableError(RuntimeError):
    """Required model could not load or produce a valid inference result."""
