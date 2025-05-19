"""
Manages the generation of prompts for the LLM, including initial prompts
and prompts for retrying based on errors or failed assertions.
"""
import inspect
import json  # For pretty printing schema if it's a dict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from .context_manager import LLMCallContext


def _format_type_hint(hint: Any) -> str:
    """Formats a type hint into a string representation for the prompt."""
    if hint is inspect.Parameter.empty or hint is inspect.Signature.empty:
        return "Any"  # Default for empty annotations
    if hasattr(hint, "__name__"):
        return hint.__name__
    if hasattr(hint, "__forward_arg__"):  # For Pydantic models used as forward refs
        return hint.__forward_arg__

    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None)
    if origin and args:
        try:
            origin_name = origin.__name__
        except AttributeError:
            origin_name = str(origin).split("[")[0]
            if "typing." in origin_name:
                origin_name = origin_name.replace("typing.", "")

        formatted_args = ", ".join([_format_type_hint(arg) for arg in args])
        return f"{origin_name}[{formatted_args}]"
    return str(hint)


def _generate_function_signature_string(func_definition: Callable, output_model: Optional[Type[BaseModel]]) -> str:
    """Generates a string representation of the function signature for the prompt."""
    sig = inspect.signature(func_definition)
    params_str = ", ".join([f"{name}: {_format_type_hint(p.annotation)}" for name, p in sig.parameters.items()])

    return_annotation_str = "Any"
    if output_model:
        return_annotation_str = output_model.__name__
    elif sig.return_annotation is not inspect.Signature.empty:
        return_annotation_str = _format_type_hint(sig.return_annotation)

    return f"def {func_definition.__name__}({params_str}) -> {return_annotation_str}:"


def generate_initial_prompt(
    func_definition: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output_model: Optional[Type[BaseModel]],
    call_context: 'LLMCallContext',
    post_hooks: Optional[List[Callable]] = None,
) -> str:
    """
    Generates the initial prompt for the LLM.
    """
    func_sig_str = _generate_function_signature_string(func_definition, output_model)
    docstring = inspect.getdoc(func_definition) or "No functional description provided."

    prompt_lines = [
        "You are a highly capable AI. Your task is to write a Python code snippet that computes the result for the Python function described below.",
        "Analyze the function signature, docstring, input arguments, and any specified output model to understand the required logic.",
        "The Python code you generate MUST define a variable named `llm_output` and assign the final computed result to it.",
        "This `llm_output` variable should be an instance of the specified output model, or a dictionary that can be directly validated into it.",
        "If no specific output model is given, `llm_output` should match the function's return type hint.",
        "Your code should be self-contained if possible, but you can assume the input arguments to the original function are available in the execution scope.",
        "The following modules are pre-imported and available for you to use directly: `re` (for regular expressions), `json` (for JSON processing), and `datetime` (for date/time operations). You do not need to import them.",
        "Do NOT include the function definition itself, only the body/logic to compute `llm_output`.",
        "Do NOT add any explanatory text, comments, or markdown formatting before or after the Python code block.",
    ]

    prompt_lines.extend(
        [
            f"The final `llm_output` variable MUST be compatible with the Pydantic model: '{output_model.__name__}'."
            if output_model
            else f"The final `llm_output` variable MUST be compatible with the return type hint: '{_format_type_hint(inspect.signature(func_definition).return_annotation)}'.",
        ]
    )

    if output_model:  # Keep schema information for the LLM to construct the output correctly
        prompt_lines.extend([f"The Pydantic model '{output_model.__name__}' schema for `llm_output` is as follows:"])
        try:
            schema_data = output_model.model_json_schema()
            prompt_lines.append(json.dumps(schema_data, indent=2))
        except AttributeError:  # Should be model_json_schema for Pydantic v2
            prompt_lines.append(
                f"(Could not automatically generate Pydantic schema for {output_model.__name__}. "
                f"Please infer structure from its name and any description in the function's docstring.)"
            )

    prompt_lines.extend(
        [
            "\n--- Function Definition ---",
            func_sig_str,
            '"""',  # Start of docstring block
            docstring,
            '"""',  # End of docstring block
        ]
    )

    if post_hooks:
        prompt_lines.append("\n--- Post-execution Assertion Hooks ---")
        prompt_lines.append(
            "After your code generates `llm_output`, the following assertion functions will be executed."
        )
        prompt_lines.append("Your `llm_output` MUST satisfy these assertions. Review their logic carefully:")
        for hook in post_hooks:
            hook_name = getattr(hook, "__name__", "unnamed_hook")
            try:
                source = inspect.getsource(hook)
                prompt_lines.append(f"\nAssertion Hook: `{hook_name}`")
                prompt_lines.append("```python")
                prompt_lines.append(source.strip())
                prompt_lines.append("```")
            except (TypeError, OSError):
                prompt_lines.append(f"\nAssertion Hook: `{hook_name}` (Source code not retrievable)")
        prompt_lines.append("---")

    # Current arguments are now added below, so this placeholder comment is removed.
    if args or kwargs:
        prompt_lines.append("\n--- Current Call Arguments (for context and to fulfill the request) ---")
        sig_params = inspect.signature(func_definition).parameters
        arg_names = list(sig_params.keys())
        for i, arg_val in enumerate(args):
            param_name = arg_names[i]
            param_type = sig_params[param_name].annotation
            prompt_lines.append(f"  Argument '{param_name}' ({_format_type_hint(param_type)}): {repr(arg_val)}")
            # If the argument is a Pydantic model, also include its schema
            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                try:
                    schema_data = param_type.model_json_schema()
                    prompt_lines.append(f"    Schema for '{param_name}' ({param_type.__name__}):")
                    prompt_lines.append(json.dumps(schema_data, indent=2))
                except AttributeError:
                    prompt_lines.append(f"    (Could not automatically generate Pydantic schema for {param_name}.)")

        for k, v_val in kwargs.items():
            param_type = sig_params[k].annotation
            prompt_lines.append(f"  Argument '{k}' ({_format_type_hint(param_type)}): {repr(v_val)}")
            # If the argument is a Pydantic model, also include its schema
            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                try:
                    schema_data = param_type.model_json_schema()
                    prompt_lines.append(f"    Schema for '{k}' ({param_type.__name__}):")
                    prompt_lines.append(json.dumps(schema_data, indent=2))
                except AttributeError:
                    prompt_lines.append(f"    (Could not automatically generate Pydantic schema for {k}.)")

    prompt_lines.append("\n--- Your Python Code Snippet ---")
    prompt_lines.append("Provide ONLY the Python code as described above:")

    return "\n".join(prompt_lines)


def generate_retry_prompt(
    func_definition: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output_model: Optional[Type[BaseModel]],
    call_context: 'LLMCallContext',
    post_hooks: Optional[List[Callable]] = None,
) -> str:
    """
    Generates a prompt for retrying after a failure.
    """
    initial_prompt_part = generate_initial_prompt(
        func_definition, args, kwargs, output_model, call_context, post_hooks
    )

    error_history_lines = ["\n\n--- Previous Attempt(s) Feedback ---"]
    attempts_history = call_context.get_attempts_history() if hasattr(call_context, "get_attempts_history") else []

    if not attempts_history:
        error_history_lines.append(
            "No detailed error history available for this retry cycle, but a previous attempt failed."
        )
    else:
        for i, attempt_log in enumerate(attempts_history):
            error_history_lines.append(f"\nAttempt {i + 1} Details:")

            if attempt_log.get("llm_response"):
                # Using repr() to better handle newlines and special chars in LLM response for display
                llm_response_summary = repr(attempt_log["llm_response"][:1000])
                if len(attempt_log["llm_response"]) > 1000:
                    llm_response_summary += "... (truncated)"
                error_history_lines.append(f"  Your raw response was: {llm_response_summary}")

            if attempt_log.get("processed_output") is not None:
                processed_output_summary = repr(str(attempt_log["processed_output"])[:1000])
                if len(str(attempt_log["processed_output"])) > 1000:
                    processed_output_summary += "... (truncated)"
                error_history_lines.append(f"  Interpreted output: {processed_output_summary}")

            error_info = attempt_log.get("error")
            if error_info and isinstance(error_info, dict):
                error_history_lines.append(f"  Failure Reason:")
                error_type = error_info.get("type", "Error")
                error_message = str(error_info.get("message", "No specific error message."))
                error_history_lines.append(f"    Type: {error_type}")
                error_history_lines.append(f"    Details: {error_message}")
                if error_info.get("hook_name"):
                    error_history_lines.append(f"    Failed in assertion hook: {error_info['hook_name']}")
                if error_info.get("failed_output_value"):
                    error_history_lines.append(
                        f"    Your output that failed the assertion: {error_info['failed_output_value']}"
                    )
                if error_info.get("hook_source_code"):
                    error_history_lines.append(f"    Source code of the failing assertion hook:")
                    error_history_lines.append(f"    ```python")
                    error_history_lines.append(f"    {error_info['hook_source_code'].strip()}")
                    error_history_lines.append(f"    ```")
            elif error_info:
                error_history_lines.append(f"  Failure Reason: {repr(str(error_info))[:1000]}")

    error_history_lines.append("\n--- Instructions for Your Next Attempt ---")
    error_history_lines.append(
        "Carefully review the feedback from the LATEST failed attempt, including any errors from your previously generated code. Identify the cause of the error."
    )
    error_history_lines.append(
        "Then, re-generate your Python code snippet to correct the issue(s) and satisfy all requirements from the original function definition."
    )
    error_history_lines.append(
        "Ensure your code defines `llm_output` correctly and that its value will conform to the required output structure/type."
    )

    full_retry_prompt = initial_prompt_part + "\n".join(error_history_lines)
    return full_retry_prompt


# Placeholder for a more sophisticated templating engine if needed (e.g., Jinja2)
# For now, f-strings and string concatenation are used.
