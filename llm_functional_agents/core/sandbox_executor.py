"""
Provides mechanisms for safely executing LLM-generated Python code.

WARNING: Executing arbitrary code is inherently risky. The sandboxing 
mechanisms provided here aim to mitigate risks but may not be foolproof against 
determined malicious actors, especially in a shared environment without further 
OS-level hardening (like cgroups, namespaces, seccomp profiles for the worker processes).

Consider more robust sandboxing (e.g., Docker, gVisor, Firecracker, or Pyodide for WASM-based isolation) 
for production systems handling highly untrusted code.

For improved Python-specific sandboxing, consider integrating the `RestrictedPython` library.
`RestrictedPython` can compile and execute code with restricted access to builtins and modules,
preventing access to unsafe operations like file system access or network calls directly from
the LLM-generated code. Example conceptual integration points are noted below.
"""

import datetime  # Added datetime
import io
import json
import multiprocessing
import re  # Pre-import safe modules
import resource  # For setting resource limits (Unix-specific)
import sys
import time
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple

from ..exceptions import SandboxExecutionError

# Default resource limits for the sandboxed process
DEFAULT_CPU_TIME_LIMIT_SECONDS = 5  # Max CPU time
DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # Max virtual memory (e.g., 256MB)
DEFAULT_WALL_TIME_LIMIT_SECONDS = 10  # Max wall clock time for the subprocess call


def _execute_code_in_sandbox_process(
    code_to_execute: str, input_args: Optional[Dict[str, Any]], pipe_conn: Connection, cpu_limit: int, mem_limit: int
):
    """
    Target function for the sandboxed subprocess.
    Sets resource limits and executes the code.
    Sends (stdout, stderr, result, exception_str) back via pipe.
    """
    resource_limit_warning: Optional[str] = None
    # Attempt to set resource limits (OS-dependent, primarily Unix)
    try:
        # CPU time limit (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        # Virtual memory limit (bytes)
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        # Note: Other limits like RLIMIT_FSIZE (file size), RLIMIT_NPROC (num processes)
        # could also be considered for more comprehensive sandboxing.
    except Exception as e:
        # Non-fatal if limits can't be set on a particular OS (e.g., Windows)
        # However, this reduces the security of the sandbox on such systems.
        resource_limit_warning = f"Sandbox Warning: Could not set resource limits: {e}"

    # Prepare execution environment
    # Restricted globals: Only allow a minimal set of builtins.
    # This is a basic measure and can be bypassed by sophisticated code.
    # More advanced sandboxing would involve more aggressive techniques (e.g., restrictedpython library).
    # CONCEPTUAL: If using RestrictedPython, this section would change significantly.
    # You would compile the code using RestrictedPython.compile_restricted first,
    # and then exec the compiled code object. The `restricted_globals` and `restricted_locals`
    # would be built according to RestrictedPython's requirements.
    restricted_globals = {
        "__builtins__": {
            k: v
            for k, v in __builtins__.items()
            if k
            in {
                "print",
                "len",
                "range",
                "list",
                "dict",
                "tuple",
                "str",
                "int",
                "float",
                "bool",
                "None",
                "True",
                "False",
                "abs",
                "all",
                "any",
                "bin",
                "callable",
                "chr",
                "complex",
                "delattr",
                "dir",
                "divmod",
                "enumerate",
                "filter",
                "format",
                "getattr",
                "hasattr",
                "hash",
                "hex",
                "id",
                "isinstance",
                "issubclass",
                "iter",
                "map",
                "max",
                "min",
                "next",
                "oct",
                "ord",
                "pow",
                "repr",
                "reversed",
                "round",
                "setattr",
                "slice",
                "sorted",
                "sum",
                "vars",
                "zip",
                "Exception",
                "ValueError",
                "TypeError",
                "AssertionError",
                # Add other safe builtins as needed
            }
        },
        "re": re,  # Make the 're' module available by name
        "json": json,  # Make the 'json' module available by name
        "datetime": datetime,  # Make the datetime module available
        # Provide input arguments to the executed code if any
        **(input_args or {}),
    }
    # Code will be executed with its own local scope
    local_scope = {}

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = io.StringIO()
    redirected_stderr = io.StringIO()
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr

    if resource_limit_warning:
        print(resource_limit_warning, file=sys.stderr)

    result: Any = None
    exception_str: Optional[str] = None

    try:
        # Execute the code. The LLM should be prompted to produce code that might assign
        # its final result to a variable named `llm_output` for extraction, or handle output
        # via print statements if that's the desired interaction pattern.
        
        # CONCEPTUAL: If using RestrictedPython, `code_to_execute` would be pre-compiled.
        # Example: 
        # from RestrictedPython import compile_restricted
        # from RestrictedPython.PrintCollector import PrintCollector
        # from RestrictedPython.Guards import safe_builtins, full_write_guard
        #
        # restricted_builtins = safe_builtins.copy()
        # restricted_builtins['_print_'] = PrintCollector # To capture prints
        # # restricted_builtins['_getattr_'] = ... (custom getattr guard)
        # # restricted_builtins['_write_'] = full_write_guard # Or a more restrictive one
        #
        # compiled_code = compile_restricted(code_to_execute, filename='<llm_generated_code>', mode='exec')
        # exec(compiled_code, {'__builtins__': restricted_builtins, **(input_args or {})}, local_scope)
        # captured_prints = restricted_builtins['_print_'].get_value()
        # stdout_val += captured_prints

        exec(code_to_execute, restricted_globals, local_scope)
        result = local_scope.get("llm_output")  # Convention for LLM to set its result
    except MemoryError as e_mem:
        exception_str = f"MemoryError (likely due to resource limit): {e_mem}"
    except RuntimeError as e_cpu:  # CPU time limit often raises RuntimeError
        exception_str = f"RuntimeError (likely due to CPU time limit): {e_cpu}"
    except Exception as e_exec:
        import traceback

        exception_str = f"ExecutionError: {type(e_exec).__name__}: {e_exec}\nTraceback:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        redirected_stdout.flush()
        redirected_stderr.flush()
        stdout_val = redirected_stdout.getvalue()
        stderr_val = redirected_stderr.getvalue()

        # Debug print to original stderr of child process
        print(f"ChildProcessDebug: stderr_val from StringIO is {repr(stderr_val)}", file=old_stderr)
        old_stderr.flush()

        try:
            pipe_conn.send((stdout_val, stderr_val, result, exception_str))
        except Exception as e_pipe:
            # If pipe communication fails, log to original stderr (this process's stderr)
            print(f"Sandbox Error: Failed to send results via pipe: {e_pipe}", file=sys.stderr)
            # Attempt to send a minimal error indication if possible
            try:
                pipe_conn.send((None, str(e_pipe), None, f"PipeSendError: {e_pipe}"))
            except:  # nosec B110
                pass  # Avoid nested exceptions if pipe is truly broken
        finally:
            pipe_conn.close()


def execute_in_sandbox(
    code_to_execute: str,
    input_args: Optional[Dict[str, Any]] = None,
    cpu_limit_secs: int = DEFAULT_CPU_TIME_LIMIT_SECONDS,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    wall_time_limit_secs: int = DEFAULT_WALL_TIME_LIMIT_SECONDS,
) -> Tuple[str, str, Any, Optional[str]]:
    """
    Executes the given Python code string in a sandboxed subprocess.

    Args:
        code_to_execute: The Python code string to execute.
        input_args: A dictionary of arguments to be made available to the executing code.
                    The code can access these as global variables.
        cpu_limit_secs: CPU time limit for the subprocess.
        memory_limit_bytes: Virtual memory limit for the subprocess.
        wall_time_limit_secs: Wall clock time limit for waiting on the subprocess.

    Returns:
        A tuple: (stdout_str, stderr_str, result, exception_str)
        - stdout_str: Captured standard output from the executed code.
        - stderr_str: Captured standard error from the executed code.
        - result: The value of a variable named 'llm_output' in the executed code's
                  local scope, or None if not set.
        - exception_str: A string representation of any exception that occurred during
                         execution, or None if no exception.

    Raises:
        SandboxExecutionError: If the sandbox process fails to start, times out,
                               or has other critical issues.
    """
    parent_conn, child_conn = multiprocessing.Pipe()

    process = multiprocessing.Process(
        target=_execute_code_in_sandbox_process,
        args=(code_to_execute, input_args, child_conn, cpu_limit_secs, memory_limit_bytes),
    )

    stdout_str = ""
    stderr_str = ""
    result_val: Any = None
    exception_val: Optional[str] = None

    try:
        process.start()
        # parent_conn should wait for data from the child process
        if parent_conn.poll(timeout=wall_time_limit_secs):  # Wait for data with timeout
            if parent_conn.closed:
                raise SandboxExecutionError("Sandbox process pipe closed unexpectedly before sending data.")
            received_data = parent_conn.recv()
            stdout_str, stderr_str, result_val, exception_val = received_data
        else:
            # Timeout occurred before child sent data
            process.terminate()  # Terminate the process if it's still running
            process.join(timeout=1)  # Attempt to join gracefully
            if process.is_alive():
                process.kill()  # Force kill if terminate + join failed
                process.join()  # Wait for kill
            raise SandboxExecutionError(
                f"Sandbox execution timed out after {wall_time_limit_secs} seconds. "
                f"Code: {code_to_execute[:200]}..."
            )

    except multiprocessing.ProcessError as e_proc:
        exception_val = f"Sandbox ProcessError: {e_proc}"
        raise SandboxExecutionError(f"Failed to manage sandbox process: {e_proc}") from e_proc
    except EOFError as e_eof:  # Pipe closed unexpectedly
        exception_val = f"Sandbox EOFError: Pipe closed prematurely. {e_eof}"
        # This might indicate the child process crashed hard or was killed by OS due to limits
        # before it could send anything or a proper error.
        # Check stderr from the process if possible (though usually captured by child)
        raise SandboxExecutionError(f"Pipe closed prematurely: {e_eof}. Child process might have crashed.") from e_eof
    except Exception as e_parent:  # Other errors in the parent process orchestration
        exception_val = f"Sandbox ParentError: {e_parent}"
        if process.is_alive():
            try:
                process.kill()
                process.join(timeout=0.5)
            except Exception as e_kill:
                print(f"Sandbox: Error during cleanup kill: {e_kill}", file=sys.stderr)
        raise SandboxExecutionError(f"Error during sandbox orchestration: {e_parent}") from e_parent
    finally:
        if process.is_alive():
            try:
                process.kill()  # Ensure process is killed if anything went wrong
                process.join(timeout=0.5)
            except Exception as e_final_kill:
                print(f"Sandbox: Error during final cleanup kill: {e_final_kill}", file=sys.stderr)
        parent_conn.close()
        child_conn.close()  # Though child should have closed it already

    return stdout_str or "", stderr_str or "", result_val, exception_val
