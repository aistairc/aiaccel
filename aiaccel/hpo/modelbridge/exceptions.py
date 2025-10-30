"""Domain specific exceptions for the modelbridge pipeline."""


class ModelBridgeError(RuntimeError):
    """Base exception for modelbridge failures."""


class ValidationError(ModelBridgeError):
    """Raised when configuration or user input is invalid."""


class ExecutionError(ModelBridgeError):
    """Raised when a trial evaluation fails to complete."""


__all__ = ["ModelBridgeError", "ValidationError", "ExecutionError"]
