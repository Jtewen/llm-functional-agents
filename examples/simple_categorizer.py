# examples/simple_categorizer.py (New File)
from typing import Optional  # Added for Optional type hint

from pydantic import BaseModel, Field

from llm_functional_agents import (  # LLMCallContext might not be used directly here, but good for consistency
    LLMCallContext,
    llm_func,
)
from llm_functional_agents.config import configure
from llm_functional_agents.exceptions import MaxRetriesExceededError, ValidationFailedError

# Ensure an LLM backend is configured (e.g., via .env or programmatically)
# It's good practice to call configure() if you want to be explicit or override .env
# For example, to use OpenRouter if it's not the default or .env isn't set for it:
# configure(default_llm_backend_id="default_openrouter")


class TextInput(BaseModel):
    text: str = Field(..., description="The text to categorize.")


class CategoryOutput(BaseModel):
    category: str = Field(..., description="The determined category for the text.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")


ALLOWED_CATEGORIES = ["general_inquiry", "technical_support", "billing_question", "positive_feedback"]


# Post-hook assertion
def check_category(output: CategoryOutput, *, data: TextInput):  # Added * for keyword-only 'data'
    assert (
        output.category in ALLOWED_CATEGORIES
    ), f"Category '{output.category}' is not one of the allowed categories: {ALLOWED_CATEGORIES}"


@llm_func(output_model=CategoryOutput, post_hooks=[check_category])
def categorize_text(data: TextInput) -> CategoryOutput:
    """
    Analyzes the input 'data.text' and determines its category.
    The category MUST be one of: general_inquiry, technical_support, billing_question, positive_feedback.
    Optionally, provide a confidence score (a float between 0.0 and 1.0) for the categorization.

    The LLM should generate Python code that assigns the result to `llm_output`.
    Example for `llm_output` if the text is about a login problem:
    llm_output = {
        "category": "technical_support",
        "confidence": 0.85
    }

    If the text is "I love this app!", example `llm_output`:
    llm_output = {
        "category": "positive_feedback",
        "confidence": 0.99
    }
    """
    # This body is not executed when the LLM is active.
    # It's a placeholder for type checking and direct calls if the agent mechanism isn't used.
    raise NotImplementedError(
        "This function's logic is intended to be fulfilled by LLM-generated code \n"
        "based on the docstring, signature, and provided 'data' argument."
    )


def main():
    sample_texts = [
        "I'm having trouble logging into my account on your website.",
        "What are your business hours?",
        "I love this new feature, it works great!",
        "I was charged twice for my last order, can you help?",
        "This is confusing.",
    ]

    # You might want to configure the LLM backend once, if not relying on .env or defaults
    # try:
    #     configure(default_llm_backend_id="default_openrouter")
    # except Exception as e:
    #     print(f"Configuration error: {e}. Ensure your .env file or config.py is set up.")
    #     return

    for i, sample_text in enumerate(sample_texts):
        print(f"--- Example {i+1} ---")
        print(f'Input text: "{sample_text}"')

        input_data = TextInput(text=sample_text)

        try:
            print("\nCalling categorize_text functional agent...")
            result: CategoryOutput = categorize_text(data=input_data)
            print("\nSuccessfully categorized text:")
            print(f"  Category: {result.category}")
            if result.confidence is not None:
                print(f"  Confidence: {result.confidence:.2f}")

        except ValidationFailedError as ve:
            print(f"\nValidation Failed after retries: {ve}")
            context = ve.last_error_context
            if context:
                print("  Last error context details:")
                print(f"    Function: {context.func_name}")
                print(f"    Attempts: {context.current_attempt_number}/{context.max_retries + 1}")
                for i, attempt_log in enumerate(context.get_attempts_history()):
                    print(f"      Attempt {i + 1} Log:")
                    if attempt_log.get("llm_response"):
                        print(f"        LLM Generated Code:\n'''{attempt_log['llm_response']}'''")
                    if attempt_log.get("error"):
                        print(f"        Error: {attempt_log['error'].get('message', 'N/A')}")
        except MaxRetriesExceededError as mre:
            print(f"\nMax Retries Exceeded: {mre}")
            if mre.last_error:
                print(f"  Last error type: {type(mre.last_error).__name__}")
                print(f"  Last error details: {mre.last_error}")
            final_context = mre.final_llm_call_context
            if final_context:
                print("  Final LLM Call Context details (includes all attempts):")
                print(f"    Function: {final_context.func_name}")
                print(f"    Total Attempts: {final_context.current_attempt_number}/{final_context.max_retries + 1}")
                for i, attempt_log in enumerate(final_context.get_attempts_history()):
                    print(f"      Attempt {i + 1} Log:")
                    if attempt_log.get("llm_response"):
                        print(f"        LLM Generated Code:\n'''{attempt_log['llm_response']}'''")
                    if attempt_log.get("error"):
                        print(f"        Error: {attempt_log['error'].get('message', 'N/A')}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
        print("---------------------\n")


if __name__ == "__main__":
    # To run: Ensure your .env file is set up with an API key (e.g., OPENROUTER_API_KEY)
    # Then, from the project root directory:
    # python -m examples.simple_categorizer
    main()
