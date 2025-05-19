"""
Core execution logic for functional agents.
Handles the retry loop, pre/post hooks, LLM calls, and validation.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import logging

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ..config import get_retry_policy, get_sandbox_limits  # Added get_sandbox_limits
from ..exceptions import (  # ConfigurationError # Not directly raised here but used by imported functions
    LLMBackendError,
    MaxRetriesExceededError,
    SandboxExecutionError,
    ValidationFailedError,
)
from ..utils.context_manager import LLMCallContext  # To be implemented
from ..utils.prompts import generate_initial_prompt, generate_retry_prompt  # To be implemented
from .llm_backends import get_llm_backend  # To be implemented
from .sandbox_executor import execute_in_sandbox  # Added for code execution

logger = logging.getLogger(__name__)

def execute_functional_agent_call(
    func_definition: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output_model: Optional[Type[BaseModel]],
    pre_hooks: List[Callable],
    post_hooks: List[Callable],  # Assertion hooks
    llm_backend_id: Optional[str],
    retry_policy_id: Optional[str],
) -> Any:
    """
    Manages the lifecycle of a single call to an @llm_func decorated function.
    This is the heart of the library's execution logic.
    """
    effective_retry_policy_id = retry_policy_id or "default"
    retry_policy_config = get_retry_policy(effective_retry_policy_id)
    max_retries = retry_policy_config.get("max_retries", 3)

    call_context = LLMCallContext(func_name=func_definition.__name__, max_retries=max_retries)
    call_context.set_initial_args(args, kwargs)

    current_args, current_kwargs = args, kwargs  # These might be modified by pre_hooks if that feature is added
    for hook in pre_hooks:
        try:
            # Current assumption: pre_hooks take original args/kwargs and raise on failure.
            # They don't modify args/kwargs for the LLM call itself, but could be adapted to.
            hook(*current_args, **current_kwargs)
        except Exception as e:
            # Consider a specific pre-hook failure exception for clarity
            raise ValidationFailedError(f"Pre-processing hook {hook.__name__} failed: {e}") from e

    attempt = 0
    last_exception: Optional[Exception] = None

    while attempt <= max_retries:
        call_context.new_attempt()

        try:
            # 2. Prepare LLM call
            llm_backend_instance = get_llm_backend(llm_backend_id)  # Gets instance from llm_backends.py
            
            logger.info(
                "Preparing LLM call for %s (Attempt %s/%s)", 
                func_definition.__name__, 
                call_context.current_attempt_number, 
                max_retries + 1
            )

            if attempt == 0:
                prompt = generate_initial_prompt(
                    func_definition, current_args, current_kwargs, output_model, call_context, post_hooks
                )
            else:
                # Pass the call_context so the retry prompt can include error history
                prompt = generate_retry_prompt(
                    func_definition, current_args, current_kwargs, output_model, call_context, post_hooks
                )

            call_context.add_prompt(prompt)
            logger.debug(
                "Prompt for %s (Attempt %s):\n%s", 
                func_definition.__name__, 
                call_context.current_attempt_number,
                prompt
            )

            # 3. Invoke LLM Backend
            raw_llm_response_content = llm_backend_instance.invoke(prompt)
            call_context.add_llm_response(raw_llm_response_content)

            # 4. Process LLM Response - Now involves sandbox execution
            # raw_llm_response_content is expected to be Python code.
            code_to_execute = raw_llm_response_content

            # Strip markdown code block fences if present
            if code_to_execute.strip().startswith("```python"):
                code_to_execute = code_to_execute.strip()[len("```python") :].strip()
                if code_to_execute.endswith("```"):
                    code_to_execute = code_to_execute[: -len("```")]
            elif code_to_execute.strip().startswith("```"):
                code_to_execute = code_to_execute.strip()[len("```") :].strip()
                if code_to_execute.endswith("```"):
                    code_to_execute = code_to_execute[: -len("```")]
            code_to_execute = code_to_execute.strip()  # Ensure leading/trailing whitespace is removed

            # --- REMOVE TEMPORARY DEBUG PRINT ---
            # print(
            #     f"--- LLM Generated Code for {func_definition.__name__} (Attempt {call_context.current_attempt_number}) ---"
            # )
            # print(code_to_execute)
            # print("--------------------------------------")
            # --- END TEMPORARY DEBUG PRINT ---

            # +++ ADD PROPER LOGGING FOR GENERATED CODE +++
            logger.debug(
                "LLM generated code for %s (Attempt %s):\n%s", 
                func_definition.__name__, 
                call_context.current_attempt_number, 
                code_to_execute
            )
            # +++ END PROPER LOGGING +++

            # Prepare input arguments for the sandbox. These are the original function args.
            # The generated code can access these by their original names.
            sandbox_input_args = {}
            sig_params = inspect.signature(func_definition).parameters
            arg_names = list(sig_params.keys())
            for i, arg_val in enumerate(args):
                sandbox_input_args[arg_names[i]] = arg_val
            sandbox_input_args.update(kwargs)

            # Get sandbox limits from config
            # Assuming retry_policy_id can also be used to identify a sandbox_limits policy for consistency
            # Or, a separate sandbox_policy_id could be introduced to @llm_func decorator
            sandbox_limits_config = get_sandbox_limits(policy_id=effective_retry_policy_id) 
            # Convert MB to Bytes for memory limit
            memory_limit_bytes = sandbox_limits_config.get("memory_limit_mb", 256) * 1024 * 1024

            # Execute the LLM-generated code in the sandbox
            stdout, stderr, sandboxed_result, sandbox_err_str = execute_in_sandbox(
                code_to_execute,
                input_args=sandbox_input_args,
                cpu_limit_secs=sandbox_limits_config.get("cpu_time_seconds", 5),
                memory_limit_bytes=int(memory_limit_bytes), # Ensure it's int
                wall_time_limit_secs=sandbox_limits_config.get("wall_time_seconds", 10)
            )
            call_context.add_sandbox_log(
                code_to_execute, stdout, stderr, sandboxed_result, sandbox_err_str
            )  # New context method

            if sandbox_err_str:
                # If sandbox itself had an error (e.g. timeout, resource limit, or code execution error)
                # This includes SyntaxError, NameError from the generated code, etc.
                call_context.add_error(
                    SandboxExecutionError(sandbox_err_str, traceback_str=sandbox_err_str), "sandbox_execution"
                )
                raise SandboxExecutionError(
                    f"Error executing LLM-generated code: {sandbox_err_str}", traceback_str=sandbox_err_str
                )

            # The convention is that sandboxed_result is the value of 'llm_output'
            llm_result_from_sandbox = sandboxed_result

            processed_output: Any
            if output_model:
                try:
                    # llm_result_from_sandbox should be a dict or an object that Pydantic can validate
                    if isinstance(llm_result_from_sandbox, dict):
                        processed_output = output_model(**llm_result_from_sandbox)
                    elif isinstance(llm_result_from_sandbox, output_model):
                        processed_output = llm_result_from_sandbox  # Already an instance
                    elif (
                        llm_result_from_sandbox is None and output_model is None
                    ):  # Explicitly allow None if output_model is None (though covered by `not output_model` path)
                        processed_output = None
                    elif llm_result_from_sandbox is None:  # LLM might return None explicitly
                        # Check if output_model allows None (e.g. Optional[SomeModel] or Union[SomeModel, None])
                        # This is a simplified check; Pydantic handles this validation well.
                        # Forcing it through model_validate will give better errors.
                        processed_output = output_model.model_validate(None)
                    else:
                        # Attempt to validate directly if it's not a dict (e.g. basic types for simple models)
                        processed_output = output_model.model_validate(llm_result_from_sandbox)

                except PydanticValidationError as e_pydantic:
                    error_msg = f"LLM output from sandbox failed Pydantic validation for {output_model.__name__}: {e_pydantic}. Sandbox output: {str(llm_result_from_sandbox)[:200]}"
                    # Adding the current call_context to the exception itself
                    current_validation_error = ValidationFailedError(error_msg, last_error_context=call_context)
                    call_context.add_error(current_validation_error, "pydantic_validation")
                    raise current_validation_error from e_pydantic
                except Exception as e_val:  # Other validation/conversion errors
                    error_msg = f"Failed to convert/validate sandbox output for {output_model.__name__}: {e_val}. Sandbox output: {str(llm_result_from_sandbox)[:200]}"
                    # Adding the current call_context to the exception itself
                    current_conversion_error = ValidationFailedError(error_msg, last_error_context=call_context)
                    call_context.add_error(current_conversion_error, "output_conversion")
                    raise current_conversion_error from e_val
            else:
                # If no output_model, the direct result from sandbox is used.
                processed_output = llm_result_from_sandbox

            call_context.add_processed_output(processed_output)

            # 5. Run post-condition assertion hooks
            for hook in post_hooks:
                try:
                    # Assertion hook signature: hook(output, *original_args, **original_kwargs)
                    hook(processed_output, *args, **kwargs)  # Pass original args/kwargs for context
                except AssertionError as e_assert:
                    hook_source_code = "<Source code not available>"
                    try:
                        hook_source_code = inspect.getsource(hook)
                    except (TypeError, OSError):
                        pass  # Keep default if source can't be fetched

                    call_context.add_error(
                        e_assert,
                        error_type="assertion_failure",
                        hook_name=hook.__name__,
                        hook_source_code=hook_source_code,
                        failed_output_value=processed_output,
                    )
                    failed_assertion_detail = f"Assertion failed in {hook.__name__}: {e_assert}"
                    # Adding the current call_context to the exception itself
                    current_assertion_error = ValidationFailedError(failed_assertion_detail, last_error_context=call_context)
                    raise current_assertion_error from e_assert

            # If all assertions pass, return the result
            call_context.set_success()
            logger.info("Functional agent %s completed successfully after %s attempt(s).", func_definition.__name__, call_context.current_attempt_number)
            return processed_output

        except (ValidationFailedError, SandboxExecutionError, LLMBackendError) as e_functional:
            last_exception = e_functional
            logger.warning(
                "Attempt %s for %s failed: %s - %s", 
                call_context.current_attempt_number, 
                func_definition.__name__, 
                type(e_functional).__name__, 
                e_functional
            )
            # Ensure error is added to context if not already (some might be added at raise point)
            # Simplified: error is added when specific exception is caught or here for generic
            current_error_type_for_context = e_functional.__class__.__name__.lower()
            if isinstance(e_functional, ValidationFailedError):
                current_error_type_for_context = "validation_failed" # More specific for context
            elif isinstance(e_functional, SandboxExecutionError):
                current_error_type_for_context = "sandbox_execution"
            elif isinstance(e_functional, LLMBackendError):
                current_error_type_for_context = "llm_backend_error"
            
            if not call_context.get_last_error() or call_context.get_last_error().get("message") != str(e_functional):
                 call_context.add_error(e_functional, error_type=current_error_type_for_context)


            # Check if we should retry this specific error type
            should_retry_this_error = True # Default to true if not specified in policy
            if isinstance(e_functional, ValidationFailedError):
                should_retry_this_error = retry_policy_config.get("retry_on_validation_failure", True)
            elif isinstance(e_functional, SandboxExecutionError):
                should_retry_this_error = retry_policy_config.get("retry_on_sandbox_error", True)
            elif isinstance(e_functional, LLMBackendError):
                should_retry_this_error = retry_policy_config.get("retry_on_backend_error", False) # Default False for backend

            if not should_retry_this_error or attempt >= max_retries: # Check attempt count with >=
                raise MaxRetriesExceededError(
                    f"Max {max_retries} retries reached for {func_definition.__name__} or policy prevents retry for this error. Last error: {last_exception}",
                    last_error=last_exception, # Keep last_error
                    final_llm_call_context=call_context # Add the full context
                ) from last_exception
            # Optional: log retry attempt here for functional error if continuing

        except Exception as e_unexpected:  # Catch any other unexpected errors during the process
            last_exception = e_unexpected
            logger.error(
                "Unexpected error on attempt %s for %s: %s - %s", 
                call_context.current_attempt_number, 
                func_definition.__name__, 
                type(e_unexpected).__name__, 
                e_unexpected, 
                exc_info=True # Include traceback information for unexpected errors
            )
            # Ensure error is added to context if not already
            if not call_context.get_last_error() or call_context.get_last_error().get("message") != str(e_unexpected):
                call_context.add_error(e_unexpected, error_type="unexpected_error")
            
            # For unexpected errors, we typically retry unless it's the last attempt.
            if attempt >= max_retries: # Check attempt count with >=
                raise MaxRetriesExceededError(
                    f"Max {max_retries} retries reached for {func_definition.__name__}. Last unexpected error: {last_exception}",
                    last_error=last_exception, # Keep last_error
                    final_llm_call_context=call_context # Add the full context
                ) from last_exception
            # Optional: log retry attempt here for unexpected error if continuing

        attempt += 1

    # This line should ideally not be reached if MaxRetriesExceededError is raised correctly within the loop.
    # However, as a final safeguard:
    raise MaxRetriesExceededError(
        f"Exited retry loop unexpectedly for {func_definition.__name__} after {attempt -1} attempts. Last error: {last_exception}",
        last_error=last_exception, # Keep last_error
        final_llm_call_context=call_context # Add the full context
    )
