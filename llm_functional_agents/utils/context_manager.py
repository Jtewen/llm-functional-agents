"""
Manages the context and state for an individual call to an @llm_func,
including retry attempts, error history, and intermediate outputs.
"""
import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict


class AttemptLogEntry(TypedDict, total=False):
    attempt_number: int
    timestamp: str
    prompt: Optional[str]
    llm_response: Optional[str]
    # Sandbox execution details
    sandbox_code: Optional[str]
    sandbox_stdout: Optional[str]
    sandbox_stderr: Optional[str]
    sandbox_result: Optional[Any]
    sandbox_exception: Optional[str]
    processed_output: Optional[Any]
    error: Optional[
        Dict[str, Any]
    ]  # e.g., {'type': 'ValidationFailed', 'message': '...', 'hook_name': '...', 'hook_source': '...', 'failed_output': '...'}
    # Add other relevant fields like tokens_used, latency_ms, etc. if available


class LLMCallContext:
    """
    Stores and manages the state for a single invocation of an LLM-driven function,
    including its retry attempts and error history.
    """

    def __init__(self, func_name: str, max_retries: int):
        self.func_name: str = func_name
        self.max_retries: int = max_retries
        self.current_attempt_number: int = 0
        self.start_time: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        self.end_time: Optional[datetime.datetime] = None
        self.is_success: bool = False

        self.initial_args: Optional[Tuple[Any, ...]] = None
        self.initial_kwargs: Optional[Dict[str, Any]] = None

        self._attempts_history: List[AttemptLogEntry] = []

    def set_initial_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Records the initial arguments passed to the function."""
        self.initial_args = args
        self.initial_kwargs = kwargs

    def new_attempt(self) -> None:
        """Increments the attempt counter and prepares for a new attempt."""
        self.current_attempt_number += 1
        self._attempts_history.append(
            {
                "attempt_number": self.current_attempt_number,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    def _current_attempt_log(self) -> Optional[AttemptLogEntry]:
        """Helper to get the log entry for the current attempt."""
        if not self._attempts_history or self._attempts_history[-1]["attempt_number"] != self.current_attempt_number:
            # This should not happen if new_attempt() is called correctly
            return None
        return self._attempts_history[-1]

    def add_prompt(self, prompt: str) -> None:
        """Adds the generated prompt for the current attempt."""
        log_entry = self._current_attempt_log()
        if log_entry:
            log_entry["prompt"] = prompt

    def add_llm_response(self, raw_response: str) -> None:
        """Adds the raw response from the LLM for the current attempt."""
        log_entry = self._current_attempt_log()
        if log_entry:
            log_entry["llm_response"] = raw_response

    def add_sandbox_log(self, code: str, stdout: str, stderr: str, result: Any, exception_str: Optional[str]) -> None:
        """Adds the details from a sandbox execution attempt."""
        log_entry = self._current_attempt_log()
        if log_entry:
            log_entry["sandbox_code"] = code
            log_entry["sandbox_stdout"] = stdout
            log_entry["sandbox_stderr"] = stderr
            try:
                log_entry["sandbox_result"] = repr(result) if result is not None else None
            except Exception:
                log_entry["sandbox_result"] = "<Sandbox result not serializable for logging>"
            if exception_str:
                log_entry["sandbox_exception"] = exception_str

    def add_processed_output(self, output: Any) -> None:
        """Adds the processed/validated output for the current attempt."""
        log_entry = self._current_attempt_log()
        if log_entry:
            # Be careful about storing large objects here; consider summarizing or referencing
            try:
                # Attempt to get a serializable representation
                log_entry["processed_output"] = repr(output)
            except Exception:
                log_entry["processed_output"] = "<Output not serializable for logging>"

    def add_error(
        self,
        error: Exception,
        error_type: Optional[str] = None,
        hook_name: Optional[str] = None,
        hook_source_code: Optional[str] = None,
        failed_output_value: Optional[Any] = None,
    ) -> None:
        """Adds error information for the current attempt."""
        log_entry = self._current_attempt_log()
        if log_entry:
            error_details: Dict[str, Any] = {
                "type": error_type or error.__class__.__name__,
                "message": str(error),
            }
            if hook_name:
                error_details["hook_name"] = hook_name
            if hook_source_code:
                error_details["hook_source_code"] = hook_source_code
            if failed_output_value is not None:
                try:
                    error_details["failed_output_value"] = repr(failed_output_value)
                except Exception:
                    error_details["failed_output_value"] = "<Failed output value not serializable for logging>"

            # For specific custom exceptions, extract more details
            if hasattr(error, "assertion_details") and isinstance(getattr(error, "assertion_details"), dict):
                error_details["assertion_details"] = getattr(error, "assertion_details")
            if hasattr(error, "traceback_str") and isinstance(getattr(error, "traceback_str"), str):
                error_details["traceback"] = getattr(error, "traceback_str")

            log_entry["error"] = error_details

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Returns the error info from the most recent attempt, if any."""
        if self._attempts_history and "error" in self._attempts_history[-1]:
            return self._attempts_history[-1]["error"]
        return None

    def set_success(self) -> None:
        """Marks the call as successful and records end time."""
        self.is_success = True
        self.end_time = datetime.datetime.now(datetime.timezone.utc)

    def get_attempts_history(self) -> List[AttemptLogEntry]:
        """Returns the history of all attempts."""
        return self._attempts_history

    def is_first_attempt(self) -> bool:
        return self.current_attempt_number == 1

    # Add other methods as needed, e.g., to get total duration, etc.
