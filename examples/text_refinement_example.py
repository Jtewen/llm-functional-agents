from typing import Tuple

from pydantic import BaseModel, Field, validator

from llm_functional_agents import (  # Corrected import
    LLMCallContext,
    MaxRetriesExceededError,
    ValidationFailedError,
    llm_func,
)
from llm_functional_agents.config import configure  # Allow configuration

# Optional: Configure LLM backend if not using default or .env settings
# configure(default_llm_backend_id="your_chosen_backend_id") # e.g., "default_openrouter" or "default_openai"


class RefineTextToLimitInput(BaseModel):
    """Input model for the smart text rewriter."""

    original_text: str = Field(..., description="The original text to be refined.")
    max_length: int = Field(..., gt=0, description="The maximum desired character length for the refined text.")


class RefineTextToLimitOutput(BaseModel):
    """Output model for the smart text rewriter."""

    original_text: str
    refined_text: str
    original_length: int
    refined_length: int
    is_within_limit: bool

    @validator("refined_text")
    def refined_text_not_empty_if_original_not_empty(cls, v, values):
        if "original_text" in values and len(values["original_text"].strip()) > 0 and len(v.strip()) == 0:
            raise ValueError("Refined text cannot be empty if original text was not empty.")
        return v


# --- Assertion Hook (Post-execution Check) ---
def check_refinement_constraints(
    output: RefineTextToLimitOutput,
    *,  # Make arguments after this keyword-only
    data: RefineTextToLimitInput,  # Expect 'data' as a keyword argument
) -> None:
    """
    Checks if the refined text meets the specified length constraints
    and was appropriately shortened if necessary.
    """
    # The 'data' argument from smart_rewriter is now directly available.

    assert (
        output.is_within_limit
    ), f"LLM claims refined text is within limit, but calculation shows is_within_limit={output.is_within_limit} (refined_length: {output.refined_length}, max_length: {data.max_length})."

    assert (
        output.refined_length <= data.max_length
    ), f"Refined text ({output.refined_length} chars) exceeds the maximum allowed length of {data.max_length} chars."

    if output.original_length > data.max_length:
        assert output.refined_length < output.original_length, (
            f"Original text ({output.original_length} chars) exceeded max length ({data.max_length} chars), "
            f"but refined text ({output.refined_length} chars) is not shorter."
        )

    # The Pydantic validator on RefineTextToLimitOutput already checks for empty refined text
    # if original was not empty. We could add an explicit assertion here too if desired for a different error message.
    # assert len(output.refined_text.strip()) > 0 if len(output.original_text.strip()) > 0 else True, \
    #     "Refined text is empty, but original text was not."


@llm_func(output_model=RefineTextToLimitOutput, post_hooks=[check_refinement_constraints])
def smart_rewriter(data: RefineTextToLimitInput) -> RefineTextToLimitOutput:
    """
    The primary task is to intelligently rewrite the 'data.original_text' to be more concise.
    The goal is for the 'refined_text' to be within the 'data.max_length' (maximum character limit)
    while preserving the core meaning and essential information of the original.

    You (the LLM) should:
    1.  Perform the text rewriting using your language understanding capabilities.
    2.  The Python code you generate should then construct the `llm_output` variable.
        This variable must be an instance of `RefineTextToLimitOutput` or a dictionary
        conforming to its schema. The values for the fields should be:
        {
            "original_text": data.original_text,
            "refined_text": "YOUR_REFINED_TEXT_HERE",
            "original_length": len(data.original_text),
            "refined_length": len("YOUR_REFINED_TEXT_HERE"),
            "is_within_limit": (len("YOUR_REFINED_TEXT_HERE") <= data.max_length)
        }

    The input 'data' object (an instance of RefineTextToLimitInput) makes 'data.original_text'
    and 'data.max_length' available by name in the execution scope of the Python code you provide.
    Ensure the Python code snippet you provide only defines `llm_output` and assigns the
    above-described dictionary or `RefineTextToLimitOutput` instance to it.
    Do not include any other statements or explanations outside this assignment.
    For example, if data.original_text is "This is a very long sentence that needs to be much shorter."
    and data.max_length is 50, a good refined_text might be "This long sentence needs shortening."
    and your code would look like:

    llm_output = {
        "original_text": data.original_text,
        "refined_text": "This long sentence needs shortening.",
        "original_length": len(data.original_text),
        "refined_length": len("This long sentence needs shortening."),
        "is_within_limit": len("This long sentence needs shortening.") <= data.max_length
    }
    """
    # This actual Python body is NOT directly executed by the LLM mechanism in typical use.
    # It serves as a fallback, for type checking, or if you called this function
    # directly in Python without the LLM agent system being active.
    # The core idea of @llm_func is that the LLM provides the operative logic.
    raise NotImplementedError(
        "This function's logic is intended to be fulfilled by LLM-generated code. "
        "The LLM should generate code to perform the rewriting and packaging of results "
        "based on the docstring, signature, and provided 'data' argument."
    )


def run_example(text_to_refine: str, char_limit: int):
    print(f"--- Task: Refine Text to <{char_limit} Chars ---")
    print(f'Original Text ({len(text_to_refine)} chars): "{text_to_refine}"')

    input_payload = RefineTextToLimitInput(original_text=text_to_refine, max_length=char_limit)

    try:
        print("\nCalling smart_rewriter functional agent...")
        # Ensure OPENROUTER_API_KEY (or your chosen provider's key) is in your .env
        # or configure the LLM backend appropriately.
        # Example: configure(default_llm_backend_id="default_openrouter")

        result_output: RefineTextToLimitOutput = smart_rewriter(data=input_payload)

        print("\nSuccessfully refined text:")
        print(f'  Original ({result_output.original_length} chars): "{result_output.original_text}"')
        print(f'  Refined ({result_output.refined_length} chars): "{result_output.refined_text}"')
        print(f"  Within {input_payload.max_length} char limit: {result_output.is_within_limit}")

        if not result_output.is_within_limit:
            print(f"  WARNING: Refined text is NOT within the {input_payload.max_length} character limit!")
        if (
            result_output.refined_length >= result_output.original_length
            and result_output.original_length > input_payload.max_length
        ):
            print(
                f"  WARNING: Refined text was not made shorter than the original, though original exceeded the limit."
            )

    except ValidationFailedError as ve:
        print(f"\nValidation Failed after retries: {ve}")
        # ve.last_error_context is the LLMCallContext object
        context = ve.last_error_context # Access the formalized attribute
        if context: # Check if context is not None
            print("Last error context details:")
            print(f"  Function: {context.func_name}")
            print(
                f"  Attempts: {context.current_attempt_number}/{context.max_retries + 1}"
            )
            # For more detail, you can inspect:
            # context.get_current_attempt_log()
            # context.get_attempts_history()
            current_attempt_log = context.get_current_attempt_log() # Use the new context variable
            if current_attempt_log and current_attempt_log.get("error"):
                print(f"  Last error message: {current_attempt_log['error'].get('message', 'N/A')}")

    except MaxRetriesExceededError as mre:
        print(f"\nMax Retries Exceeded: {mre}")
        if mre.last_error:
            print(f"  Last error type: {type(mre.last_error).__name__}")
            print(f"  Last error details: {mre.last_error}")
        # Access the formalized attribute
        final_context = mre.final_llm_call_context 
        if final_context: # Check if context is not None
            print("  Final LLM Call Context details:")
            print(f"    Function: {final_context.func_name}")
            print(f"    Attempts: {final_context.current_attempt_number}/{final_context.max_retries + 1}")
            final_attempt_log = final_context.get_current_attempt_log()
            if final_attempt_log and final_attempt_log.get("error"):
                print(f"    Last error message in final attempt: {final_attempt_log['error'].get('message', 'N/A')}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    print("--- End of Task ---")


def main():
    # Example 1: Text that needs significant shortening
    text1 = "This is an exceptionally long and rather verbose sentence that absolutely needs to be drastically shortened to meet the stringent character limit."
    limit1 = 50
    run_example(text1, limit1)

    print("\n" + "=" * 50 + "\n")

    # Example 2: Text that is already short but could be very slightly rephrased
    text2 = "This is short."
    limit2 = 15
    run_example(text2, limit2)

    print("\n" + "=" * 50 + "\n")

    # Example 3: Text that is already well within the limit
    text3 = "Perfectly fine."
    limit3 = 50
    run_example(text3, limit3)

    print("\n" + "=" * 50 + "\n")

    # Example 4: Text that is very hard to shorten to the limit without losing meaning
    text4 = "The quick brown fox jumps over the lazy dog."  # 43 chars
    limit4 = 20
    run_example(text4, limit4)

    print("\nText refinement demo finished.")


if __name__ == "__main__":
    # To run from the project root (llm-functional-agents/):
    # python -m examples.text_refinement_example
    # Ensure your .env file with API keys (e.g., OPENROUTER_API_KEY) is in the project root,
    # or you have configured an LLM backend.
    main()
