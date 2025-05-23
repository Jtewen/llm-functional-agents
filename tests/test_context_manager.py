import unittest
import datetime
import json # For checking serializable representations if needed, though repr is mostly used.

from llm_functional_agents.utils.context_manager import LLMCallContext, AttemptLogEntry

# --- Dummy Exception with extra attributes for testing add_error ---
class CustomErrorForTesting(Exception):
    def __init__(self, msg, assertion_details=None, traceback_str=None):
        super().__init__(msg)
        self.assertion_details = assertion_details
        self.traceback_str = traceback_str

# --- Non-serializable object for testing logging ---
class NonSerializableObject:
    def __repr__(self):
        raise TypeError("This object is not serializable for repr!")

    def __str__(self):
        raise TypeError("This object is not serializable for str!")


class TestLLMCallContext(unittest.TestCase):

    def test_initialization(self):
        func_name = "test_func"
        max_retries = 3
        context = LLMCallContext(func_name=func_name, max_retries=max_retries)

        self.assertEqual(context.func_name, func_name)
        self.assertEqual(context.max_retries, max_retries)
        self.assertEqual(context.current_attempt_number, 0)
        self.assertIsInstance(context.start_time, datetime.datetime)
        self.assertIsNone(context.end_time)
        self.assertFalse(context.is_success)
        self.assertEqual(context._attempts_history, [])
        self.assertIsNone(context.initial_args)
        self.assertIsNone(context.initial_kwargs)
        # is_first_attempt should be true if current_attempt_number is 0 or 1.
        # The implementation is `self.current_attempt_number == 1`.
        # Let's test this directly in attempt management.

    def test_set_initial_args(self):
        context = LLMCallContext("set_args_func", 2)
        test_args = (1, "arg2")
        test_kwargs = {"key1": "value1", "key2": 100}
        context.set_initial_args(test_args, test_kwargs)

        self.assertEqual(context.initial_args, test_args)
        self.assertEqual(context.initial_kwargs, test_kwargs)

    def test_attempt_management_and_is_first_attempt(self):
        context = LLMCallContext("attempt_func", 3)
        
        # Before any new_attempt call, current_attempt_number is 0.
        # is_first_attempt compares current_attempt_number == 1.
        self.assertFalse(context.is_first_attempt()) 

        # First attempt
        context.new_attempt()
        self.assertEqual(context.current_attempt_number, 1)
        self.assertEqual(len(context._attempts_history), 1)
        first_attempt_log = context._attempts_history[0]
        self.assertEqual(first_attempt_log["attempt_number"], 1)
        self.assertTrue(isinstance(first_attempt_log["timestamp"], str))
        self.assertTrue(context.is_first_attempt())

        # Second attempt
        context.new_attempt()
        self.assertEqual(context.current_attempt_number, 2)
        self.assertEqual(len(context._attempts_history), 2)
        second_attempt_log = context._attempts_history[1]
        self.assertEqual(second_attempt_log["attempt_number"], 2)
        self.assertFalse(context.is_first_attempt())

    def test_add_prompt_to_current_attempt(self):
        context = LLMCallContext("prompt_func", 1)
        context.new_attempt()
        prompt_text = "This is a test prompt."
        context.add_prompt(prompt_text)
        
        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("prompt"), prompt_text)

    def test_add_llm_response_to_current_attempt(self):
        context = LLMCallContext("response_func", 1)
        context.new_attempt()
        response_text = "This is the LLM's raw response."
        context.add_llm_response(response_text)

        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("llm_response"), response_text)

    def test_add_sandbox_log_basic(self):
        context = LLMCallContext("sandbox_func", 1)
        context.new_attempt()
        code, stdout, stderr, result = "print('hi')", "hi\n", "", "None" # result is repr'd
        context.add_sandbox_log(code, stdout, stderr, result, None)

        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("sandbox_code"), code)
        self.assertEqual(current_log.get("sandbox_stdout"), stdout)
        self.assertEqual(current_log.get("sandbox_stderr"), stderr)
        self.assertEqual(current_log.get("sandbox_result"), repr(result)) # result is stored as repr
        self.assertIsNone(current_log.get("sandbox_exception"))

    def test_add_sandbox_log_with_exception(self):
        context = LLMCallContext("sandbox_exc_func", 1)
        context.new_attempt()
        exception_str = "ZeroDivisionError: division by zero"
        context.add_sandbox_log("1/0", "", "Error output", None, exception_str)

        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("sandbox_exception"), exception_str)

    def test_add_sandbox_log_with_non_serializable_result(self):
        context = LLMCallContext("sandbox_nonserial_func", 1)
        context.new_attempt()
        non_serializable = NonSerializableObject()
        context.add_sandbox_log("code", "stdout", "stderr", non_serializable, None)
        
        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("sandbox_result"), "<Sandbox result not serializable for logging>")

    def test_add_processed_output_basic(self):
        context = LLMCallContext("proc_out_func", 1)
        context.new_attempt()
        output_data = {"key": "value", "number": 123}
        context.add_processed_output(output_data)

        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("processed_output"), repr(output_data))

    def test_add_processed_output_non_serializable(self):
        context = LLMCallContext("proc_out_nonserial_func", 1)
        context.new_attempt()
        non_serializable = NonSerializableObject()
        context.add_processed_output(non_serializable)

        current_log = context._attempts_history[-1]
        self.assertEqual(current_log.get("processed_output"), "<Output not serializable for logging>")
        
    def test_add_error_standard_exception(self):
        context = LLMCallContext("error_std_func", 1)
        context.new_attempt()
        error = ValueError("Standard error message")
        context.add_error(error)

        current_log = context._attempts_history[-1]
        error_log = current_log.get("error")
        self.assertIsNotNone(error_log)
        self.assertEqual(error_log.get("type"), "ValueError")
        self.assertEqual(error_log.get("message"), "Standard error message")

    def test_add_error_with_custom_details(self):
        context = LLMCallContext("error_custom_func", 1)
        context.new_attempt()
        error = Exception("Base error")
        custom_type = "CustomErrorType"
        hook_name = "my_post_hook"
        hook_source = "assert output > 0"
        failed_val = -1
        
        context.add_error(error, error_type=custom_type, hook_name=hook_name, hook_source_code=hook_source, failed_output_value=failed_val)
        
        current_log = context._attempts_history[-1]
        error_log = current_log.get("error")
        self.assertIsNotNone(error_log)
        self.assertEqual(error_log.get("type"), custom_type)
        self.assertEqual(error_log.get("message"), "Base error")
        self.assertEqual(error_log.get("hook_name"), hook_name)
        self.assertEqual(error_log.get("hook_source_code"), hook_source)
        self.assertEqual(error_log.get("failed_output_value"), repr(failed_val))

    def test_add_error_with_non_serializable_failed_value(self):
        context = LLMCallContext("error_nonserial_val_func", 1)
        context.new_attempt()
        error = Exception("Error with non-serializable failed value")
        non_serializable = NonSerializableObject()
        context.add_error(error, failed_output_value=non_serializable)

        current_log = context._attempts_history[-1]
        error_log = current_log.get("error")
        self.assertIsNotNone(error_log)
        self.assertEqual(error_log.get("failed_output_value"), "<Failed output value not serializable for logging>")


    def test_add_error_with_exception_attributes(self):
        context = LLMCallContext("error_attrs_func", 1)
        context.new_attempt()
        assertion_details = {"field": "x", "reason": "too small"}
        traceback_str = "Traceback (most recent call last):\n  File \"<stdin>\", line 1, in <module>\nValueError: sample error"
        error = CustomErrorForTesting("Error with attributes", assertion_details=assertion_details, traceback_str=traceback_str)
        
        context.add_error(error)
        
        current_log = context._attempts_history[-1]
        error_log = current_log.get("error")
        self.assertIsNotNone(error_log)
        self.assertEqual(error_log.get("type"), "CustomErrorForTesting")
        self.assertEqual(error_log.get("message"), "Error with attributes")
        self.assertEqual(error_log.get("assertion_details"), assertion_details)
        self.assertEqual(error_log.get("traceback"), traceback_str)

    def test_get_last_error(self):
        context = LLMCallContext("get_last_err_func", 3)
        self.assertIsNone(context.get_last_error()) # No attempts, no error

        context.new_attempt() # Attempt 1: No error
        self.assertIsNone(context.get_last_error())

        context.new_attempt() # Attempt 2: Add an error
        error1_obj = ValueError("First error")
        context.add_error(error1_obj)
        last_error_1 = context.get_last_error()
        self.assertIsNotNone(last_error_1)
        self.assertEqual(last_error_1.get("message"), "First error")

        context.new_attempt() # Attempt 3: No error initially
        self.assertIsNone(context.get_last_error()) # Should be None as current attempt has no error yet

        error2_obj = TypeError("Second error on third attempt") # Add error to attempt 3
        context.add_error(error2_obj)
        last_error_2 = context.get_last_error()
        self.assertIsNotNone(last_error_2)
        self.assertEqual(last_error_2.get("message"), "Second error on third attempt")
        self.assertEqual(last_error_2.get("type"), "TypeError")


    def test_set_success(self):
        context = LLMCallContext("success_func", 1)
        context.set_success()

        self.assertTrue(context.is_success)
        self.assertIsInstance(context.end_time, datetime.datetime)
        # Ensure end_time is later than or equal to start_time
        self.assertTrue(context.end_time >= context.start_time)

    def test_get_attempts_history(self):
        context = LLMCallContext("history_func", 2)
        self.assertEqual(context.get_attempts_history(), [])

        context.new_attempt()
        context.add_prompt("Prompt 1")
        history_after_1 = context.get_attempts_history()
        self.assertEqual(len(history_after_1), 1)
        self.assertEqual(history_after_1[0]["attempt_number"], 1)
        self.assertEqual(history_after_1[0]["prompt"], "Prompt 1")

        context.new_attempt()
        context.add_prompt("Prompt 2")
        context.add_llm_response("Response 2")
        history_after_2 = context.get_attempts_history()
        self.assertEqual(len(history_after_2), 2)
        self.assertEqual(history_after_2[1]["attempt_number"], 2)
        self.assertEqual(history_after_2[1]["prompt"], "Prompt 2")
        self.assertEqual(history_after_2[1]["llm_response"], "Response 2")

        # Check that it returns a copy, not the internal list directly (optional, but good practice)
        # Modifying the returned list should not affect the internal one.
        returned_history = context.get_attempts_history()
        returned_history.append(AttemptLogEntry(attempt_number=3, timestamp="dummy")) # type: ignore
        self.assertEqual(len(context._attempts_history), 2) # Internal should remain unchanged


if __name__ == "__main__":
    unittest.main()
```
