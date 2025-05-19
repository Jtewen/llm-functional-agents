import unittest
import sys
import os
from unittest import mock # Added for @patch

# Add the project root to the Python path to allow importing from llm_functional_agents
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_functional_agents.core.sandbox_executor import execute_in_sandbox, SandboxExecutionError
from llm_functional_agents.core.sandbox_executor import (
    DEFAULT_CPU_TIME_LIMIT_SECONDS,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_WALL_TIME_LIMIT_SECONDS
)

class TestSandboxExecutor(unittest.TestCase):

    def test_simple_execution_with_output_variable(self):
        code = "print('Hello from sandbox!'); llm_output = 42"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "Hello from sandbox!\n")
        self.assertEqual(stderr, "")
        self.assertEqual(res, 42)
        self.assertIsNone(err)

    def test_os_import_restricted(self):
        code = "import os; print(os.getcwd()); llm_output = 'Should fail or be restricted'"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "") # os.getcwd() should not print
        self.assertIn("ImportError", err) # __import__ is not available
        self.assertIn("__import__ not found", err)
        self.assertIsNone(res)

    def test_input_attempt_restricted(self):
        code = "llm_output = input('This should not hang: ')"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "")
        self.assertIn("NameError", err) # input is not defined in restricted builtins
        self.assertIn("'input' is not defined", err)
        self.assertIsNone(res)

    def test_memory_limit(self):
        # Attempt to allocate 200MB, limit to 100MB
        code = "a = ' ' * (200 * 1024 * 1024); llm_output = len(a)"
        # Lower memory limit significantly for testing to ensure it triggers
        # Note: Actual memory usage can be tricky to predict perfectly for 'exec'
        # This test might be OS/environment dependent if limits are not strictly enforced
        # or if the overhead of the process itself is close to the limit.
        # Let's use a very small limit that should definitely be exceeded by the string allocation.
        # However, setting it *too* low might cause the process to fail before even running the code.
        # The default in sandbox_executor is 256MB, let's try with 50MB.
        low_mem_limit = 50 * 1024 * 1024 
        
        # It's possible the error manifests as a general SandboxExecutionError if the process
        # is killed by the OS before our internal error handling in the child can report MemoryError
        # or if the pipe breaks.
        try:
            stdout, stderr, res, err = execute_in_sandbox(code, memory_limit_bytes=low_mem_limit)
            # Depending on OS and precise timing, MemoryError might be in err, or stderr
            # For Unix, setrlimit should cause a MemoryError within the child process.
            self.assertTrue("MemoryError" in str(err) or "Sandbox process pipe closed unexpectedly" in str(err) or "RuntimeError" in str(err), f"Unexpected error: {err}, stderr: {stderr}")
            self.assertIsNone(res)
        except SandboxExecutionError as e:
            # This can happen if the process is killed hard by OS due to memory limits
            # or if the pipe breaks because the child died.
            self.assertTrue("timed out" not in str(e), "Should be memory error, not timeout")
            print(f"Caught SandboxExecutionError (expected for memory limit): {e}")
        except Exception as e:
            self.fail(f"Unexpected exception type during memory limit test: {type(e).__name__}: {e}")


    def test_cpu_time_limit(self):
        code = "for i in range(10**8): pass; llm_output = 'done'"
        # Note: The wall_time_limit_secs must be greater than cpu_limit_secs
        # for the CPU limit to be the primary cause of termination.
        try:
            stdout, stderr, res, err = execute_in_sandbox(
                code, 
                cpu_limit_secs=1, 
                wall_time_limit_secs=3
            )
            # CPU limit often raises RuntimeError in the child, caught and reported in 'err'
            self.assertIn("RuntimeError", err)
            self.assertIn("CPU time limit", err)
            self.assertIsNone(res)
        except SandboxExecutionError as e:
            # This might occur if the wall time limit is hit very closely with CPU time
            self.assertIn("timed out", str(e).lower(), "If SandboxExecutionError, it should be due to timeout closely related to CPU exhaustion")


    def test_wall_time_timeout(self):
        # Use a loop that will definitely exceed wall time if not interrupted.
        # Ensure CPU limit is higher than wall time to isolate wall time timeout.
        code = "for _ in range(10**8): pass; llm_output = 'Loop finished'"
        with self.assertRaises(SandboxExecutionError) as context:
            execute_in_sandbox(code, cpu_limit_secs=5, wall_time_limit_secs=1)
        self.assertIn("timed out after 1 seconds", str(context.exception))

    def test_unsafe_builtin_restricted(self):
        code = "llm_output = open('/etc/passwd', 'r').read()"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "")
        self.assertIn("NameError", err)
        self.assertIn("'open' is not defined", err)
        self.assertIsNone(res)

    def test_division_by_zero(self):
        code = "llm_output = 1/0"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "")
        self.assertIn("ZeroDivisionError", err)
        self.assertIsNone(res)

    def test_no_output_variable(self):
        code = "print('Only printing')"
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "Only printing\n")
        self.assertEqual(stderr, "")
        self.assertIsNone(res) # llm_output was not set
        self.assertIsNone(err)
        
    def test_empty_code_string(self):
        code = ""
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        self.assertIsNone(res)
        self.assertIsNone(err)

    def test_syntax_error_in_code(self):
        code = "llm_output = 1 / / 0" # Syntax error
        stdout, stderr, res, err = execute_in_sandbox(code)
        self.assertEqual(stdout, "")
        self.assertIn("SyntaxError", err)
        self.assertIsNone(res)

    def test_input_args_provided(self):
        code = "llm_output = x * y"
        input_args = {"x": 5, "y": 10}
        stdout, stderr, res, err = execute_in_sandbox(code, input_args=input_args)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        self.assertEqual(res, 50)
        self.assertIsNone(err)

    def test_input_args_overwriting_builtins_partially_prevented(self):
        # 'print' is a builtin we allow. If user passes 'print' in input_args,
        # it should not overwrite the builtin 'print' inside exec.
        # However, the current sandbox implementation gives input_args higher precedence
        # if they clash with the selected `__builtins__`. This is a known characteristic.
        # The test will verify the current behavior.
        code = "try:\n  print('hello')\n  llm_output = 'printed'\nexcept TypeError:\n  llm_output = 'print is not callable'"
        input_args = {"print": "not_a_function"} # Attempt to overwrite print
        
        # With current sandbox: input_args (globals) take precedence over restricted_globals['__builtins__']
        # So, print will be "not_a_function", and calling it will cause a TypeError.
        stdout, stderr, res, err = execute_in_sandbox(code, input_args=input_args)

        self.assertEqual(stdout, "") # The original print('hello') should not execute
        self.assertEqual(stderr, "")
        # The code is expected to catch the TypeError and set llm_output.
        # This confirms 'print' from input_args was used.
        self.assertEqual(res, "print is not callable") 
        self.assertIsNone(err) # No *sandbox* error, the error is handled *in* the sandboxed code.

    def test_extremely_long_stdout(self):
        # Test that a large amount of stdout doesn't break the pipe or handling.
        # Create a string that is e.g. 1MB. Pipes have buffers.
        long_string_part = "a" * 1024 
        num_repeats = 200 # 200KB of 'a's
        code = f"for i in range({num_repeats}): print('{long_string_part}')"
        
        stdout, stderr, res, err = execute_in_sandbox(code, wall_time_limit_secs=15) # Increased wall time
        
        expected_stdout_line = long_string_part + "\n"
        expected_full_stdout = expected_stdout_line * num_repeats
        
        self.assertEqual(len(stdout), len(expected_full_stdout))
        self.assertEqual(stdout, expected_full_stdout)
        self.assertEqual(stderr, "")
        self.assertIsNone(res)
        self.assertIsNone(err)

    @mock.patch('llm_functional_agents.core.sandbox_executor.resource.setrlimit')
    def test_resource_setrlimit_fails_graceful_degradation(self, mock_setrlimit):
        mock_setrlimit.side_effect = OSError("Mocked setrlimit failure")

        code = "llm_output = sum(range(10**3))" # Simple code that should run
        
        stdout, stderr, res, err = execute_in_sandbox(
            code,
            # Limits are passed but mock_setrlimit will cause them to fail to be set
            cpu_limit_secs=1, 
            memory_limit_bytes=1024*1024*100, 
            wall_time_limit_secs=5
        )
        
        self.assertIsNone(err) # No *execution* error from the sandboxed code itself.
        self.assertIsNotNone(res) # Should complete.
        self.assertEqual(res, sum(range(10**3)))
        # Check for the warning message in stderr.
        self.assertIn("Sandbox Warning: Could not set resource limits", stderr)
        self.assertIn("Mocked setrlimit failure", stderr) # Check for our specific mock error message

if __name__ == '__main__':
    unittest.main() 