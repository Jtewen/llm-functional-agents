import sys
print(f"PYTHON SYS.PATH in test_integration.py (from llm-functional-agents/tests/test_integration.py): {sys.path}\n") # DIAGNOSTIC PRINT

import unittest
from unittest import mock
import pydantic # Retaining this as it's standard and models use it
from typing import Any, Optional

from llm_functional_agents.core.llm_function import llm_func
from llm_functional_agents.core.llm_backends import LLMBackend
from llm_functional_agents.utils.context_manager import LLMCallContext # Used by prompts, good to have for context

# --- Mock LLM Backend ---
class MockLLMBackend(LLMBackend):
    def __init__(self, backend_id: str = "mock_backend_for_test", registered_config: Optional[dict] = None):
        # Temporarily configure the mock backend for super().__init__()
        from llm_functional_agents.config import configure, get_llm_backend_config
        
        self.original_config_state = None # To store what was there before
        self.backend_id_to_restore = backend_id

        try:
            # Try to get existing config to restore it later
            self.original_config_state = get_llm_backend_config(backend_id)
        except Exception:
            self.original_config_state = None # Indicates it didn't exist or was problematic

        # Configure the mock backend
        mock_specific_config = registered_config or {"type": "mock", "default_model": "mock-model"}
        configure(llm_backends={backend_id: mock_specific_config})
        
        super().__init__(backend_id=backend_id) # Now this should find its config
        
        self.last_prompt_received: Optional[str] = None
        self.response_to_return: str = "{\"result\": \"mocked LLM response\"}" 
        self.call_count: int = 0

    def _initialize_client(self) -> Any:
        return "mock_client_instance"

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt_received = prompt
        self.call_count += 1
        return self.response_to_return
    
    def restore_config(self):
        """Helper to restore config state if needed after test."""
        from llm_functional_agents.config import configure, _config_store
        if self.original_config_state is not None:
            # Restore the original configuration for this backend_id
            configure(llm_backends={self.backend_id_to_restore: self.original_config_state})
        elif self.backend_id_to_restore in _config_store.get("llm_backends", {}):
            # If it didn't exist before but was added, attempt to remove it.
            # This is crude; a proper config system would have unregister.
            try:
                del _config_store["llm_backends"][self.backend_id_to_restore]
                if not _config_store["llm_backends"]: # if dict becomes empty
                    del _config_store["llm_backends"]
            except KeyError:
                pass # Already gone or structure changed

# --- Pydantic Models for Testing ---
class MyTestOutput(pydantic.BaseModel):
    result: str

class MyFunctionInput(pydantic.BaseModel):
    value: int

# --- Test Class ---
class TestCoreIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_backend_instance = MockLLMBackend(backend_id="mock_integration_test_backend")
        
        # Patch get_llm_backend in agent_executor where it's called
        self.patcher = mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend')
        self.mock_get_llm_backend = self.patcher.start()
        self.mock_get_llm_backend.return_value = self.mock_backend_instance
    
    def tearDown(self):
        self.patcher.stop()
        # Attempt to restore config changed by MockLLMBackend instance
        if hasattr(self.mock_backend_instance, 'restore_config'):
            self.mock_backend_instance.restore_config()

    def test_simple_llm_func_with_pydantic_output(self):
        self.mock_backend_instance.response_to_return = '{\"result\": \"mocked pydantic output\"}'

        @llm_func(output_model=MyTestOutput)
        def simple_task_pydantic(instruction: str) -> MyTestOutput:
            """Generates a result based on instruction."""
            # Body not executed by mock
            raise NotImplementedError("Should be mocked")

        response_data = simple_task_pydantic("Test instruction")

        self.assertIsInstance(response_data, MyTestOutput)
        self.assertEqual(response_data.result, "mocked pydantic output")
        self.mock_get_llm_backend.assert_called_once() 
        self.assertIsNotNone(self.mock_backend_instance.last_prompt_received)
        if self.mock_backend_instance.last_prompt_received: # Check for None before assertIn
            self.assertIn("MyTestOutput", self.mock_backend_instance.last_prompt_received)
            self.assertIn("Test instruction", self.mock_backend_instance.last_prompt_received)
        self.assertEqual(self.mock_backend_instance.call_count, 1)

    def test_llm_func_with_primitive_return_and_no_output_model(self):
        # Set the mock response to be just a raw string, as llm_output will be this directly
        self.mock_backend_instance.response_to_return = "llm_output = \"raw string output from llm\""

        @llm_func() 
        def simple_task_primitive(data: str) -> str:
            """Processes some data."""
            raise NotImplementedError("Should be mocked")

        response_str = simple_task_primitive("input data")
        
        # The sandbox executes the string, which sets llm_output
        self.assertEqual(response_str, "raw string output from llm")
        self.mock_get_llm_backend.assert_called_once()
        self.assertIsNotNone(self.mock_backend_instance.last_prompt_received)
        if self.mock_backend_instance.last_prompt_received:
            self.assertIn("input data", self.mock_backend_instance.last_prompt_received)
            self.assertIn("-> str", self.mock_backend_instance.last_prompt_received) 
        self.assertEqual(self.mock_backend_instance.call_count, 1)

    def test_llm_func_with_pydantic_input_and_output(self):
        self.mock_backend_instance.response_to_return = 'llm_output = {\"result\": \"output based on pydantic input\"}'
        
        @llm_func(output_model=MyTestOutput)
        def process_pydantic_input(data: MyFunctionInput) -> MyTestOutput:
            """Processes structured input data."""
            raise NotImplementedError("Should be mocked")

        input_data = MyFunctionInput(value=123)
        response_data = process_pydantic_input(data=input_data) # Pass as keyword arg

        self.assertIsInstance(response_data, MyTestOutput)
        self.assertEqual(response_data.result, "output based on pydantic input")
        self.mock_get_llm_backend.assert_called_once()
        self.assertIsNotNone(self.mock_backend_instance.last_prompt_received)
        if self.mock_backend_instance.last_prompt_received:
            self.assertIn("MyFunctionInput", self.mock_backend_instance.last_prompt_received)
            self.assertIn("value: int", self.mock_backend_instance.last_prompt_received) 
            self.assertIn("'data': MyFunctionInput(value=123)", self.mock_backend_instance.last_prompt_received)
        self.assertEqual(self.mock_backend_instance.call_count, 1)

if __name__ == '__main__':
    unittest.main()