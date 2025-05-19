"""
LLM Functional Agents Library

Primary exports and version information.
"""

__version__ = "0.1.0"

# Attempt to import and re-export key components for the public API.
# This will be populated as the components are built.
try:
    from .core.llm_function import llm_func
except ImportError:
    llm_func = None  # Placeholder if not yet defined

try:
    from .utils.context_manager import LLMCallContext
except ImportError:
    LLMCallContext = None  # Placeholder

try:
    from .exceptions import (  # noqa: F401
        ConfigurationError,
        FunctionalAgentError,
        LLMBackendError,
        MaxRetriesExceededError,
        SandboxExecutionError,
        ValidationFailedError,
    )
except ImportError:
    # Define placeholder exceptions if the file doesn't exist yet or is empty
    class FunctionalAgentError(Exception):
        pass

    class ValidationFailedError(FunctionalAgentError):
        pass

    class SandboxExecutionError(FunctionalAgentError):
        pass

    class LLMBackendError(FunctionalAgentError):
        pass

    class MaxRetriesExceededError(FunctionalAgentError):
        pass

    class ConfigurationError(FunctionalAgentError):
        pass


__all__ = [
    "llm_func",
    "FunctionalAgentError",
    "ValidationFailedError",
    "SandboxExecutionError",
    "LLMBackendError",
    "MaxRetriesExceededError",
    "ConfigurationError",
    "LLMCallContext",
]
