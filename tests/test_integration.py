import sys
print(f"PYTHON SYS.PATH in test_integration.py (from llm-functional-agents/tests/test_integration.py): {sys.path}\n") # DIAGNOSTIC PRINT

import unittest
from unittest import mock
import pydantic 
from typing import Any, Optional, List, Dict
from copy import deepcopy # For config store management

from llm_functional_agents.core.llm_function import llm_func
from llm_functional_agents.core.llm_backends import LLMBackend, LLMBackendConfig
from llm_functional_agents.utils.context_manager import LLMCallContext
from llm_functional_agents.exceptions import LLMGenerationError, LLMOutputProcessingError
from llm_functional_agents.config import configure, _config_store, get_llm_backend_config, _get_llm_backend_config_dict


# --- Dummy Error for Testing ---
class LLMAPIError(Exception):
    """A dummy API error for testing."""
    pass

# --- Mock LLM Backend ---
class MockLLMBackend(LLMBackend):
    def __init__(self, backend_id: str = "mock_backend_for_test", registered_config: Optional[LLMBackendConfig] = None):
        # Ensure this backend_id is configured before super().__init__
        # This is a bit of a workaround for testing; normally config happens once.
        self._original_config_for_id = None
        global _config_store
        if backend_id in _config_store.get("llm_backends", {}):
             self._original_config_for_id = deepcopy(_config_store["llm_backends"][backend_id])

        # If a specific config is provided, use it. Otherwise, create a default mock one.
        if registered_config:
            # Convert Pydantic model to dict for storing in _config_store if it's not already
            if isinstance(registered_config, LLMBackendConfig):
                 current_config_dict = registered_config.model_dump(exclude_none=True)
            else: # Assuming it's already a dict
                 current_config_dict = registered_config
        else:
            # Default mock config if none provided
            current_config_dict = {"type": "mock_type", "config": {"model": "mock-model"}}
            if "llm_backends" not in _config_store:
                _config_store["llm_backends"] = {}
            _config_store["llm_backends"][backend_id] = current_config_dict


        # Ensure the backend is configured for super().__init__()
        if backend_id not in _config_store.get("llm_backends", {}):
            # This path should ideally not be hit if registered_config or default above works
            configure(llm_backends={backend_id: {"type": "mock_default_type", "config": {"model": "default-mock-model"}}})
        
        super().__init__(backend_id=backend_id)
        
        self.last_prompt_received: Optional[str] = None
        self.response_to_return: str = "{\"result\": \"mocked LLM response\"}" 
        self.call_count: int = 0
        self.raise_exception_on_invoke: Optional[Exception] = None
        self.return_malformed_json: bool = False
        self.return_error_code_snippet: bool = False


    def _initialize_client(self) -> Any:
        return "mock_client_instance"

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt_received = prompt
        self.call_count += 1
        if self.raise_exception_on_invoke:
            raise self.raise_exception_on_invoke
        if self.return_malformed_json:
            return "this is not valid json"
        if self.return_error_code_snippet:
            return "llm_output = 1 / 0" # Code that will raise ZeroDivisionError
        return self.response_to_return
    
    def restore_config_for_id(self):
        """Helper to restore config state for this specific backend_id."""
        global _config_store
        if self._original_config_for_id is not None:
            if "llm_backends" not in _config_store: # Should not happen if it was there
                _config_store["llm_backends"] = {}
            _config_store["llm_backends"][self.backend_id] = self._original_config_for_id
        elif self.backend_id in _config_store.get("llm_backends", {}):
            # If it didn't exist before (original_config_for_id is None) but was added by this instance
            try:
                del _config_store["llm_backends"][self.backend_id]
                if not _config_store["llm_backends"]:
                    del _config_store["llm_backends"] # Remove empty dict
            except KeyError:
                pass # Already gone


# --- Pydantic Models for Testing ---
class MyTestOutput(pydantic.BaseModel):
    result: str

class MyFunctionInput(pydantic.BaseModel):
    value: int

# Phase 1: Complex Pydantic Models
class NestedChildModel(pydantic.BaseModel):
    child_id: str
    child_value: Optional[int] = None

class NestedParentModel(pydantic.BaseModel):
    parent_id: str
    child_direct: NestedChildModel
    child_optional: Optional[NestedChildModel] = None
    description: str

class ListParentModel(pydantic.BaseModel):
    list_id: str
    children_list: List[NestedChildModel]
    optional_children_list: Optional[List[NestedChildModel]] = None


# --- Test Class for llm_func enhancements ---
class TestCoreIntegration(unittest.TestCase):

    def setUp(self):
        # Each test will create its own backend instance if needed, or use a shared one.
        # This default one is for tests that don't need specific backend config modification.
        self.mock_backend_instance = MockLLMBackend(backend_id="default_mock_for_core_integration")
        
        # Patch get_llm_backend in agent_executor where it's called by llm_func
        self.patcher = mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend')
        self.mock_get_llm_backend = self.patcher.start()
        self.mock_get_llm_backend.return_value = self.mock_backend_instance
    
    def tearDown(self):
        self.patcher.stop()
        if hasattr(self.mock_backend_instance, 'restore_config_for_id'):
            self.mock_backend_instance.restore_config_for_id()

    def test_simple_llm_func_with_pydantic_output(self):
        self.mock_backend_instance.response_to_return = 'llm_output = {\"result\": \"mocked pydantic output\"}'

        @llm_func(output_model=MyTestOutput)
        def simple_task_pydantic(instruction: str) -> MyTestOutput:
            """Generates a result based on instruction."""
            raise NotImplementedError("Should be mocked")

        response_data = simple_task_pydantic("Test instruction")

        self.assertIsInstance(response_data, MyTestOutput)
        self.assertEqual(response_data.result, "mocked pydantic output")
        self.mock_get_llm_backend.assert_called_once() 
        self.assertIsNotNone(self.mock_backend_instance.last_prompt_received)
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIn("MyTestOutput", prompt_content)
        self.assertIn("Test instruction", prompt_content)
        self.assertEqual(self.mock_backend_instance.call_count, 1)

    def test_llm_func_with_primitive_return_and_no_output_model(self):
        self.mock_backend_instance.response_to_return = "llm_output = \"raw string output from llm\""

        @llm_func() 
        def simple_task_primitive(data: str) -> str:
            """Processes some data."""
            raise NotImplementedError("Should be mocked")

        response_str = simple_task_primitive("input data")
        
        self.assertEqual(response_str, "raw string output from llm")
        self.mock_get_llm_backend.assert_called_once()
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIsNotNone(prompt_content)
        self.assertIn("input data", prompt_content)
        self.assertIn("-> str", prompt_content) 
        self.assertEqual(self.mock_backend_instance.call_count, 1)

    def test_llm_func_with_pydantic_input_and_output(self):
        self.mock_backend_instance.response_to_return = 'llm_output = {\"result\": \"output based on pydantic input\"}'
        
        @llm_func(output_model=MyTestOutput)
        def process_pydantic_input(data: MyFunctionInput) -> MyTestOutput:
            """Processes structured input data."""
            raise NotImplementedError("Should be mocked")

        input_data = MyFunctionInput(value=123)
        response_data = process_pydantic_input(data=input_data) 

        self.assertIsInstance(response_data, MyTestOutput)
        self.assertEqual(response_data.result, "output based on pydantic input")
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIsNotNone(prompt_content)
        self.assertIn("MyFunctionInput", prompt_content)
        self.assertIn("value: int", prompt_content) 
        self.assertIn(repr(input_data), prompt_content) # Check representation of input data
        self.assertEqual(self.mock_backend_instance.call_count, 1)

    # --- Phase 1: llm_func tests for complex Pydantic models ---
    def test_llm_func_with_nested_pydantic_input_output(self):
        child = NestedChildModel(child_id="c1", child_value=100)
        parent_input = NestedParentModel(parent_id="p1", child_direct=child, description="Test Parent Input")
        
        # Mock LLM response should be a JSON string that can be parsed into NestedParentModel
        # For llm_func, the response should be `llm_output = { ... }`
        expected_output_child = NestedChildModel(child_id="cout1", child_value=200)
        expected_output_parent_dict = {
            "parent_id": "pout1",
            "child_direct": expected_output_child.model_dump(),
            "description": "Test Parent Output"
        }
        self.mock_backend_instance.response_to_return = f"llm_output = {expected_output_parent_dict!r}"


        @llm_func(output_model=NestedParentModel)
        def process_nested_pydantic(parent_data: NestedParentModel) -> NestedParentModel:
            """Processes nested Pydantic data."""
            raise NotImplementedError("Should be mocked")

        response_data = process_nested_pydantic(parent_data=parent_input)

        self.assertIsInstance(response_data, NestedParentModel)
        self.assertEqual(response_data.parent_id, "pout1")
        self.assertEqual(response_data.description, "Test Parent Output")
        self.assertEqual(response_data.child_direct, expected_output_child)
        self.assertIsNone(response_data.child_optional) # Not provided in mock output

        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIsNotNone(prompt_content)
        self.assertIn("NestedParentModel", prompt_content) # Input type in signature
        self.assertIn(NestedParentModel.__name__, prompt_content) # Output type in prompt
        self.assertIn(repr(parent_input), prompt_content) # Check input data serialization
        # Check if schema of NestedParentModel (as input) is in prompt
        self.assertIn(NestedParentModel.model_json_schema(ref_template="#/components/schemas/{model}")["title"], prompt_content)


    def test_llm_func_with_list_of_pydantic_input_output(self):
        child1 = NestedChildModel(child_id="lc1", child_value=10)
        child2 = NestedChildModel(child_id="lc2", child_value=20)
        list_parent_input = ListParentModel(list_id="lp1", children_list=[child1, child2])

        expected_output_child1 = NestedChildModel(child_id="lout_c1", child_value=30)
        expected_output_child2 = NestedChildModel(child_id="lout_c2", child_value=40)
        expected_output_dict = {
            "list_id": "lout_p1",
            "children_list": [expected_output_child1.model_dump(), expected_output_child2.model_dump()]
        }
        self.mock_backend_instance.response_to_return = f"llm_output = {expected_output_dict!r}"

        @llm_func(output_model=ListParentModel)
        def process_list_pydantic(list_data: ListParentModel) -> ListParentModel:
            """Processes list of Pydantic data."""
            raise NotImplementedError("Should be mocked")

        response_data = process_list_pydantic(list_data=list_parent_input)

        self.assertIsInstance(response_data, ListParentModel)
        self.assertEqual(response_data.list_id, "lout_p1")
        self.assertEqual(len(response_data.children_list), 2)
        self.assertEqual(response_data.children_list[0], expected_output_child1)
        self.assertEqual(response_data.children_list[1], expected_output_child2)

        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIsNotNone(prompt_content)
        self.assertIn("ListParentModel", prompt_content) # Input type in signature
        self.assertIn(ListParentModel.__name__, prompt_content)   # Output type in prompt
        self.assertIn(repr(list_parent_input), prompt_content)  # Check input data serialization
        self.assertIn(ListParentModel.model_json_schema(ref_template="#/components/schemas/{model}")["title"], prompt_content)


    # --- Phase 1: llm_func tests for List/Dictionary return types ---
    def test_llm_func_with_list_str_return(self):
        expected_list = ["apple", "banana", "cherry"]
        self.mock_backend_instance.response_to_return = f"llm_output = {expected_list!r}"

        @llm_func()
        def get_string_list(input_text: str) -> List[str]:
            """Returns a list of strings."""
            raise NotImplementedError("Should be mocked")

        response_list = get_string_list("some fruit related query")
        self.assertEqual(response_list, expected_list)
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIn("-> List[str]", prompt_content)
        self.assertIn("some fruit related query", prompt_content)

    def test_llm_func_with_dict_str_int_return(self):
        expected_dict = {"alpha": 1, "beta": 2, "gamma": 3}
        self.mock_backend_instance.response_to_return = f"llm_output = {expected_dict!r}"

        @llm_func()
        def get_dict_str_int(input_text: str) -> Dict[str, int]:
            """Returns a dictionary of string to int."""
            raise NotImplementedError("Should be mocked")

        response_dict = get_dict_str_int("some dictionary query")
        self.assertEqual(response_dict, expected_dict)
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIn("-> Dict[str, int]", prompt_content)
        self.assertIn("some dictionary query", prompt_content)

    def test_llm_func_with_list_dict_return(self):
        expected_list_dict = [{"item": "A", "value": 10}, {"item": "B", "value": 20}]
        self.mock_backend_instance.response_to_return = f"llm_output = {expected_list_dict!r}"

        @llm_func()
        def get_list_of_dicts(input_text: str) -> List[Dict[str, Any]]:
            """Returns a list of dictionaries."""
            raise NotImplementedError("Should be mocked")

        response_list_dict = get_list_of_dicts("query for list of dicts")
        self.assertEqual(response_list_dict, expected_list_dict)
        prompt_content = self.mock_backend_instance.last_prompt_received
        self.assertIn("-> List[Dict[str, Any]]", prompt_content) # Check formatting of Any
        self.assertIn("query for list of dicts", prompt_content)


# --- Test Class for AgentExecutor features ---
class TestAgentExecutorFeatures(unittest.TestCase):
    _original_config_store_backup = None

    @classmethod
    def setUpClass(cls):
        # Backup the entire config store once before any tests in this class run
        global _config_store
        cls._original_config_store_backup = deepcopy(_config_store)

    @classmethod
    def tearDownClass(cls):
        # Restore the original config store once after all tests in this class have run
        global _config_store
        _config_store.clear()
        _config_store.update(cls._original_config_store_backup)

    def setUp(self):
        # Clear and setup specific config for each test
        global _config_store
        _config_store.clear()
        # Minimal config, tests will add to it.
        _config_store.update({"default_llm_backend_id": "default_test_backend"}) 
        
        # Instances of backends for tests to use and potentially modify
        self.error_sim_backend = MockLLMBackend(backend_id="error_sim_backend_for_test")
        self.default_backend_for_selection = MockLLMBackend(backend_id="default_test_backend", registered_config={"type": "mock_default_sel", "config": {"model": "sel-def"}})
        self.specific_backend_for_selection = MockLLMBackend(backend_id="specific_test_backend", registered_config={"type": "mock_specific_sel", "config": {"model": "sel-spec"}})

        # Patch get_llm_backend for error simulation tests where we directly control the instance
        # For backend selection tests, we might unpatch this or use a different approach.
        self.patcher = mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend')
        self.mock_get_llm_backend_for_error_tests = self.patcher.start()
        self.mock_get_llm_backend_for_error_tests.return_value = self.error_sim_backend


    def tearDown(self):
        self.patcher.stop() # Stop patcher started in setUp
        # Clean up mock backend configs added during specific tests, if any.
        # This relies on MockLLMBackend.restore_config_for_id() to clean its own ID.
        if hasattr(self.error_sim_backend, 'restore_config_for_id'):
            self.error_sim_backend.restore_config_for_id()
        if hasattr(self.default_backend_for_selection, 'restore_config_for_id'):
            self.default_backend_for_selection.restore_config_for_id()
        if hasattr(self.specific_backend_for_selection, 'restore_config_for_id'):
            self.specific_backend_for_selection.restore_config_for_id()
        
        # The class-level tearDownClass will handle restoring the global _config_store to its original state.


    # --- Phase 2: AgentExecutor - Error Handling Tests ---
    def test_agent_executor_handles_llm_api_error(self):
        self.error_sim_backend.raise_exception_on_invoke = LLMAPIError("Simulated API failure")

        @llm_func()
        def task_fails_due_to_api_error(prompt_text: str) -> str:
            raise NotImplementedError("Should be mocked")

        with self.assertRaises(LLMGenerationError) as context:
            task_fails_due_to_api_error("test")
        
        self.assertIn("Simulated API failure", str(context.exception))
        self.assertEqual(self.error_sim_backend.call_count, 1)

    def test_agent_executor_handles_malformed_json_for_pydantic(self):
        self.error_sim_backend.return_malformed_json = True

        @llm_func(output_model=MyTestOutput)
        def task_fails_due_to_bad_json(prompt_text: str) -> MyTestOutput:
            raise NotImplementedError("Should be mocked")

        with self.assertRaises(LLMOutputProcessingError) as context:
            task_fails_due_to_bad_json("test")
            
        self.assertIn("Failed to parse LLM output into Pydantic model", str(context.exception))
        self.assertIn("this is not valid json", str(context.exception)) # Original malformed output
        self.assertEqual(self.error_sim_backend.call_count, 1)

    def test_agent_executor_handles_sandbox_exec_error(self):
        self.error_sim_backend.return_error_code_snippet = True # llm_output = 1 / 0

        @llm_func()
        def task_fails_in_sandbox(prompt_text: str) -> int: # Expecting int, but will get code error
            raise NotImplementedError("Should be mocked")

        with self.assertRaises(LLMOutputProcessingError) as context:
            task_fails_in_sandbox("test")
        
        self.assertIn("Error executing LLM generated code in sandbox", str(context.exception))
        self.assertIn("division by zero", str(context.exception).lower()) # Check for underlying error
        self.assertEqual(self.error_sim_backend.call_count, 1)

    # --- Phase 2: AgentExecutor - Backend Selection Tests ---
    # For these tests, we need to unpatch `get_llm_backend` so the actual config lookup happens.
    @mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend_config', wraps=_get_llm_backend_config_dict) # Use actual config getter
    def test_uses_default_backend_when_id_not_specified(self, mock_get_conf):
        self.patcher.stop() # Stop the class-level patch of get_llm_backend

        # Configure backends: default_test_backend (set as default), specific_test_backend
        configure(
            default_llm_backend_id="default_test_backend",
            llm_backends={
                "default_test_backend": {"type": "mock_default_type", "config": {"model": "default-model-selector-test"}},
                "specific_test_backend": {"type": "mock_specific_type", "config": {"model": "specific-model-selector-test"}}
            }
        )
        # We need to ensure the LLMBackend instances used by the actual get_llm_backend are our mocks
        # So, we patch the LLMBackend class itself, or where it's instantiated.
        # For simplicity, we'll mock the `invoke` method of the specific backend instances we have.

        with mock.patch.object(self.default_backend_for_selection, 'invoke', wraps=self.default_backend_for_selection.invoke) as mock_default_invoke, \
             mock.patch.object(self.specific_backend_for_selection, 'invoke', wraps=self.specific_backend_for_selection.invoke) as mock_specific_invoke:

            # Replace actual backend construction with our instances for this test
            def side_effect_get_backend(backend_id, **kwargs):
                if backend_id == "default_test_backend":
                    return self.default_backend_for_selection
                elif backend_id == "specific_test_backend":
                    return self.specific_backend_for_selection
                raise KeyError(f"Unexpected backend_id {backend_id} in test mock")
            
            with mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend', side_effect=side_effect_get_backend):
                @llm_func()
                def task_uses_default_backend(prompt: str) -> str:
                    raise NotImplementedError("mocked")
                
                self.default_backend_for_selection.response_to_return = "llm_output = \"default_called\""
                result = task_uses_default_backend("hello")

        self.assertEqual(result, "default_called")
        mock_default_invoke.assert_called_once()
        mock_specific_invoke.assert_not_called()
        self.patcher.start() # Restart class-level patch if other tests in this class need it (though setUp/tearDown manage it)


    @mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend_config', wraps=_get_llm_backend_config_dict)
    def test_uses_specified_backend_when_id_is_given(self, mock_get_conf):
        self.patcher.stop() # Stop the class-level patch

        configure(
            default_llm_backend_id="default_test_backend",
            llm_backends={
                "default_test_backend": {"type": "mock_default_type", "config": {"model": "default-model-selector-test2"}},
                "specific_test_backend": {"type": "mock_specific_type", "config": {"model": "specific-model-selector-test2"}}
            }
        )

        with mock.patch.object(self.default_backend_for_selection, 'invoke', wraps=self.default_backend_for_selection.invoke) as mock_default_invoke, \
             mock.patch.object(self.specific_backend_for_selection, 'invoke', wraps=self.specific_backend_for_selection.invoke) as mock_specific_invoke:

            def side_effect_get_backend(backend_id, **kwargs):
                if backend_id == "default_test_backend":
                    return self.default_backend_for_selection
                elif backend_id == "specific_test_backend":
                    return self.specific_backend_for_selection
                raise KeyError(f"Unexpected backend_id {backend_id} in test mock")

            with mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend', side_effect=side_effect_get_backend):
                @llm_func(backend_id="specific_test_backend")
                def task_uses_specific_backend(prompt: str) -> str:
                    raise NotImplementedError("mocked")

                self.specific_backend_for_selection.response_to_return = "llm_output = \"specific_called\""
                result = task_uses_specific_backend("world")

        self.assertEqual(result, "specific_called")
        mock_specific_invoke.assert_called_once()
        mock_default_invoke.assert_not_called()
        self.patcher.start()


    @mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend_config', wraps=_get_llm_backend_config_dict)
    def test_raises_error_for_non_existent_backend_id(self, mock_get_conf):
        self.patcher.stop() # Stop the class-level patch

        configure(
            default_llm_backend_id="default_test_backend",
            llm_backends={
                "default_test_backend": {"type": "mock_default_type", "config": {"model": "default-model-selector-test3"}}
            }
        )
        
        # No need to mock invoke here, as get_llm_backend should fail first
        def side_effect_get_backend(backend_id, **kwargs):
            # Simulate actual behavior: get_llm_backend tries to get config, then instantiate
            # If config not found by get_llm_backend_config, it raises KeyError.
            # If get_llm_backend_config finds it, but LLMBackend init fails, that's another error.
            # Here, we assume the config lookup itself will fail.
            get_llm_backend_config(backend_id) # This will raise KeyError if not found
            # Should not reach here for "non_existent_backend"
            if backend_id == "default_test_backend": return self.default_backend_for_selection 
            # Fallback for unexpected
            raise RuntimeError(f"Test's get_llm_backend mock received unexpected id: {backend_id}")


        with mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend', side_effect=side_effect_get_backend):
            @llm_func(backend_id="non_existent_backend")
            def task_uses_non_existent_backend(prompt: str) -> str:
                raise NotImplementedError("mocked")

            with self.assertRaises(KeyError) as context: # Expecting KeyError from config lookup
                task_uses_non_existent_backend("test")
            self.assertIn("non_existent_backend", str(context.exception))
        
        self.patcher.start()


if __name__ == '__main__':
    unittest.main()