"""Custom exceptions for the llm-functional-agents library."""

from typing import Any, Dict, List, Optional

# Forward declaration for LLMCallContext to avoid circular import
# This is a common pattern when type hinting with classes defined in other modules
# that might also import from this module.
if False: # TYPE_CHECKING block
    from .utils.context_manager import LLMCallContext


class FunctionalAgentError(Exception):
    """Base exception for all errors raised by this library."""

    pass


class ValidationFailedError(FunctionalAgentError):
    """Raised when an LLM output fails post-condition assertions."""

    def __init__(self, message: str, assertion_details: dict = None, last_error_context: Optional['LLMCallContext'] = None):
        super().__init__(message)
        self.assertion_details = assertion_details or {}
        self.last_error_context: Optional['LLMCallContext'] = last_error_context


class SandboxExecutionError(FunctionalAgentError):
    """Raised when LLM-generated code fails to execute correctly in the sandbox."""

    def __init__(self, message: str, error_type: str = None, traceback_str: str = None):
        super().__init__(message)
        self.error_type = error_type
        self.traceback_str = traceback_str


class LLMBackendError(FunctionalAgentError):
    """Raised when there is an issue with the LLM backend communication (e.g., API error)."""

    def __init__(self, message: str, status_code: int = None, backend_response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.backend_response = backend_response or {}


class MaxRetriesExceededError(FunctionalAgentError):
    """Raised when an operation fails after the maximum number of retries."""

    def __init__(self, message: str, last_error: Optional[Exception] = None, final_llm_call_context: Optional['LLMCallContext'] = None):
        super().__init__(message)
        self.last_error = last_error
        self.final_llm_call_context: Optional['LLMCallContext'] = final_llm_call_context


class ConfigurationError(FunctionalAgentError):
    """Raised for configuration-related issues (e.g., missing API key, invalid setting)."""

    pass
