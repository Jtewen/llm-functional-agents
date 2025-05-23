import unittest
from unittest import mock

from llm_functional_agents.exceptions import (
    FunctionalAgentError,
    ValidationFailedError,
    SandboxExecutionError,
    LLMBackendError,
    MaxRetriesExceededError,
    ConfigurationError,
)
from llm_functional_agents.utils.context_manager import LLMCallContext
from llm_functional_agents.core.llm_function import llm_func
from llm_functional_agents.core.llm_backends import LLMBackend # For mock
from llm_functional_agents.config import configure, _config_store # For MaxRetries test
from copy import deepcopy


# --- Mock LLMCallContext for instantiation tests where None is not enough ---
class MockLLMCallContext(LLMCallContext):
    def __init__(self, func_name="mock_func", max_retries=0):
        super().__init__(func_name, max_retries)
        # Add any methods or attributes that might be accessed by the exception
        self._attempts_history = [{"attempt_number": 1, "error": {"message": "Previous error"}}]


class TestExceptionInstantiation(unittest.TestCase):

    def test_functional_agent_error(self):
        msg = "A base functional agent error occurred."
        err = FunctionalAgentError(msg)
        self.assertEqual(str(err), msg)

    def test_validation_failed_error(self):
        msg = "Validation of LLM output failed."
        details = {"field": "name", "reason": "too short"}
        mock_context = MockLLMCallContext()
        
        err = ValidationFailedError(msg, assertion_details=details, last_error_context=mock_context)
        self.assertEqual(str(err), msg)
        self.assertEqual(err.assertion_details, details)
        self.assertIs(err.last_error_context, mock_context)

        err_no_details = ValidationFailedError(msg)
        self.assertEqual(err_no_details.assertion_details, {})
        self.assertIsNone(err_no_details.last_error_context)


    def test_sandbox_execution_error(self):
        msg = "Code execution failed in sandbox."
        err_type = "NameError"
        tb_str = "Traceback: ..."
        
        err = SandboxExecutionError(msg, error_type=err_type, traceback_str=tb_str)
        self.assertEqual(str(err), msg)
        self.assertEqual(err.error_type, err_type)
        self.assertEqual(err.traceback_str, tb_str)

        err_no_details = SandboxExecutionError(msg)
        self.assertIsNone(err_no_details.error_type)
        self.assertIsNone(err_no_details.traceback_str)

    def test_llm_backend_error(self):
        msg = "LLM API returned an error."
        status = 500
        response = {"error": {"message": "Internal server error"}}
        
        err = LLMBackendError(msg, status_code=status, backend_response=response)
        self.assertEqual(str(err), msg)
        self.assertEqual(err.status_code, status)
        self.assertEqual(err.backend_response, response)

        err_no_details = LLMBackendError(msg)
        self.assertIsNone(err_no_details.status_code)
        self.assertEqual(err_no_details.backend_response, {})


    def test_max_retries_exceeded_error(self):
        msg = "Operation failed after maximum retries."
        last_err = ValueError("Last attempt failed")
        mock_context = MockLLMCallContext(max_retries=3)
        
        err = MaxRetriesExceededError(msg, last_error=last_err, final_llm_call_context=mock_context)
        self.assertEqual(str(err), msg)
        self.assertIs(err.last_error, last_err)
        self.assertIs(err.final_llm_call_context, mock_context)

        err_no_details = MaxRetriesExceededError(msg)
        self.assertIsNone(err_no_details.last_error)
        self.assertIsNone(err_no_details.final_llm_call_context)

    def test_configuration_error(self):
        msg = "Invalid configuration setting."
        err = ConfigurationError(msg)
        self.assertEqual(str(err), msg)


# --- Mock LLM Backend for MaxRetriesExceededError Test ---
class FailingMockLLMBackend(LLMBackend):
    def __init__(self, backend_id: str):
        # Minimal setup to satisfy LLMBackend constructor if it needs config
        global _config_store
        if backend_id not in _config_store.get("llm_backends", {}):
            _config_store.setdefault("llm_backends", {})[backend_id] = {"type": "mock_failing"}
        super().__init__(backend_id)
        self.call_count = 0

    def _initialize_client(self) -> str:
        return "mock_failing_client"

    def invoke(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        # This response will cause Pydantic validation to fail for MyOutputModel
        return "llm_output = {\"wrong_field\": \"this will not validate\"}" 

# --- Pydantic Model for MaxRetriesExceededError Test ---
from pydantic import BaseModel
class MyOutputModel(BaseModel):
    correct_field: str


class TestExceptionRaisingScenarios(unittest.TestCase):
    _original_config_store_backup = None

    @classmethod
    def setUpClass(cls):
        global _config_store
        cls._original_config_store_backup = deepcopy(_config_store)

    @classmethod
    def tearDownClass(cls):
        global _config_store
        _config_store.clear()
        _config_store.update(cls._original_config_store_backup)

    def setUp(self):
        global _config_store
        _config_store.clear()
        _config_store.update({
            "default_llm_backend_id": "failing_mock_backend",
            "llm_backends": {
                "failing_mock_backend": {"type": "failing_mock_type"} # type for factory
            }
        })
        self.failing_backend_instance = FailingMockLLMBackend(backend_id="failing_mock_backend")
        self.patcher = mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend')
        self.mock_get_llm_backend = self.patcher.start()
        self.mock_get_llm_backend.return_value = self.failing_backend_instance
    
    def tearDown(self):
        self.patcher.stop()

    def test_max_retries_exceeded_error_raised(self):
        max_retries_val = 1 # Configure for 1 retry (2 attempts total)

        # The llm_func decorator uses the max_retries from its own call context,
        # which is initialized by AgentExecutor.
        # The AgentExecutor's default_max_retries is 2.
        # To control this for the test, we can't easily override AgentExecutor's default.
        # Instead, we ensure the function fails consistently and check the context.
        # The max_retries parameter in @llm_func itself is what matters here.

        @llm_func(output_model=MyOutputModel, max_retries=max_retries_val)
        def consistently_failing_task(instruction: str) -> MyOutputModel:
            """This task will always fail validation."""
            # The mock LLM returns output that doesn't match MyOutputModel
            raise NotImplementedError("Should be mocked and fail validation")

        with self.assertRaises(MaxRetriesExceededError) as context:
            consistently_failing_task("Do the impossible")

        raised_exception = context.exception
        self.assertIsNotNone(raised_exception.final_llm_call_context)
        self.assertIsInstance(raised_exception.final_llm_call_context, LLMCallContext)
        
        # Total attempts = max_retries + 1 initial attempt
        self.assertEqual(raised_exception.final_llm_call_context.current_attempt_number, max_retries_val + 1)
        self.assertEqual(self.failing_backend_instance.call_count, max_retries_val + 1)
        
        # Check that the last error in the context is a ValidationFailedError (or similar)
        last_error_in_context = raised_exception.final_llm_call_context.get_last_error()
        self.assertIsNotNone(last_error_in_context)
        # The error type here will be Pydantic's ValidationError wrapped by LLMOutputProcessingError's logic
        # which then gets wrapped by MaxRetriesExceededError.
        # The direct `last_error` attribute of MaxRetriesExceededError might be an LLMOutputProcessingError.
        from llm_functional_agents.exceptions import LLMOutputProcessingError
        self.assertIsInstance(raised_exception.last_error, LLMOutputProcessingError)
        self.assertIn("Failed to parse LLM output into Pydantic model", str(raised_exception.last_error))


if __name__ == '__main__':
    unittest.main()
