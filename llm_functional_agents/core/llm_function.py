"""
Defines the @llm_func decorator and its underlying logic.
"""
import functools
import inspect
from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel

# Assuming agent_executor will have a function to call
# from .agent_executor import _execute_functional_agent_call

# Define types for hooks for clarity, even if they are just Callables for now
PreHookType = Callable[..., Any]  # Can modify args/kwargs or run checks
PostHookType = Callable[..., None]  # Typically for assertions on the result


def llm_func(
    *,  # Enforce keyword arguments for clarity
    output_model: Optional[Type[BaseModel]] = None,
    pre_hooks: Optional[List[PreHookType]] = None,
    post_hooks: Optional[List[PostHookType]] = None,  # These are the assertion hooks
    llm_backend_id: Optional[str] = None,  # Identifier for configured LLM backend
    retry_policy_id: Optional[str] = None,  # Identifier for configured retry policy
    # Potentially add: prompt_template_id, sandbox_policy_id, etc.
):
    """
    Decorator to transform a Python function definition (signature and docstring)
    into an LLM-driven functional agent.

    The decorated function's docstring is used as the primary prompt/instruction for the LLM.
    Its type hints (and `output_model` if provided) define the expected I/O structure.

    Args:
        output_model: Optional Pydantic model to validate/coerce the LLM's structured output.
        pre_hooks: Optional list of functions to run before LLM invocation.
                   Each hook receives the original function's arguments.
        post_hooks: Optional list of functions (assertions) to run after LLM invocation.
                    Each hook receives the LLM's output and original arguments.
                    Should raise AssertionError on failure.
        llm_backend_id: Specific LLM backend to use (overrides global default).
        retry_policy_id: Specific retry policy to use (overrides global default).
    """

    def decorator(func_to_wrap: Callable) -> Callable:
        # Validate that func_to_wrap is a function
        if not inspect.isfunction(func_to_wrap):
            raise TypeError("@llm_func can only decorate functions.")

        # Further introspection of func_to_wrap can happen here or in agent_executor
        # e.g., func_to_wrap.__name__, func_to_wrap.__doc__, inspect.signature(func_to_wrap)

        @functools.wraps(func_to_wrap)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # This is where the call is delegated to the agent_executor
            # For now, a direct import and call; might be refactored.
            from .agent_executor import execute_functional_agent_call  # Deferred import

            return execute_functional_agent_call(
                func_definition=func_to_wrap,
                args=args,
                kwargs=kwargs,
                output_model=output_model,
                pre_hooks=pre_hooks or [],
                post_hooks=post_hooks or [],
                llm_backend_id=llm_backend_id,
                retry_policy_id=retry_policy_id,
            )

        # Store some of the decorator's config on the wrapper if needed for introspection later
        # wrapper._llm_func_config = { ... }
        return wrapper

    return decorator
