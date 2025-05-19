import logging

from examples.extract_invoice import ExtractedInvoice, InvoiceText, extract_invoice_details  # Added for new example
from examples.normalize_address import NormalizedAddress, RawAddress, normalize_address  # Corrected import
from examples.text_refinement_example import (  # Added for new example
    RefineTextToLimitInput,
    RefineTextToLimitOutput,
    smart_rewriter,
)
from llm_functional_agents import (  # Import exceptions, add LLMCallContext
    LLMCallContext,
    MaxRetriesExceededError,
    ValidationFailedError,
    llm_func,
)
from llm_functional_agents.config import configure  # Allow configuration

# Placeholder for the actual agent execution logic - no longer needed, @llm_func handles it.
# from llm_functional_agents.core.agent_executor import execute_llm_function_call


def main():
    # --- Configure Logging --- 
    # For basic visibility during demo, setting level to DEBUG.
    # An application using this library would configure logging more globally.
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # To specifically target only this library's logs, you could do:
    # lib_logger = logging.getLogger("llm_functional_agents") # Or "llm_functional_agents.core" for more specific
    # lib_logger.setLevel(logging.DEBUG)
    # lib_logger.addHandler(logging.StreamHandler()) # To see output in console
    # --- End Logging Configuration ---

    print("Starting LLM Functional Agents Demo...\n")

    # Configure to use OpenRouter if .env is set up correctly
    # (This assumes OPENROUTER_API_KEY is in your .env)
    # The default is already OpenRouter, but this shows how to configure if needed:
    # configure(default_llm_backend_id="default_openrouter")
    # You can also select specific models for a backend in config.py or via configure()

    # Example 1: Normalize Address
    print("--- Task: Normalize Address ---")
    raw_addr_str = "123 main street, anytown, ca 90210, USA"
    # raw_addr_str_complex = "attn: john doe, po box 100, apt 4b, new york, n.y. 10001"

    print(f"Input Address: {raw_addr_str}")

    try:
        # This call will now trigger the full agent execution flow via @llm_func
        print("\nCalling normalize_address functional agent...")
        input_data = RawAddress(address_string=raw_addr_str)

        # Since execute_functional_agent_call is synchronous, but we might want async LLM calls later,
        # we'll prepare for that. For now, the call itself is blocking.
        # If your LLM client is async, execute_functional_agent_call would need to be async too.
        # Current OpenAI client is synchronous.
        result: NormalizedAddress = normalize_address(input_data)

        print("\nSuccessfully Normalized Address:")
        print(result.model_dump_json(indent=2))

    except ValidationFailedError as ve:
        print(f"\nValidation Failed after retries for address normalization: {ve}")
        context = ve.last_error_context
        if context:
            print("Last error context details:")
            print(f"  Function: {context.func_name}")
            print(f"  Attempts: {context.current_attempt_number}/{context.max_retries + 1}")
            for i, attempt_log in enumerate(context.get_attempts_history()):
                print(f"    Attempt {i + 1} Log:")
                if attempt_log.get("llm_response"):
                    print(f"      LLM Generated Code:\n'''{attempt_log['llm_response']}'''")
                if attempt_log.get("error"):
                    print(f"      Error: {attempt_log['error'].get('message', 'N/A')}")
    except MaxRetriesExceededError as mre:
        print(f"\nMax Retries Exceeded for address normalization: {mre}")
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
        print(f"\nAn unexpected error occurred during address normalization: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # --- Example 2: Extract Invoice Details ---
    print("\n\n--- Task: Extract Invoice Details ---")
    sample_invoice_text = """
    INVOICE
    Vendor: Creative Solutions Ltd.
    Invoice ID: CS-2024-105
    Issue Date: 2024-03-15
    Due Date: 2024-04-14
    Customer: Global Corp Inc.
    Attn: Procurement Dept.

    Description	Quantity	Unit Price	Total
    Web Design Services	1	$1500.00	$1500.00
    SEO Consulting	10	$100.00	$1000.00
    Content Writing (5 pages)	1	$500.00	$500.00

    Subtotal:	$3000.00
    Tax (10%):	$300.00
    TOTAL AMOUNT DUE:	$3300.00
    Currency: USD
    """
    print(f"Input Invoice Text:\n{sample_invoice_text[:300]}...")  # Print a snippet

    try:
        print("\nCalling extract_invoice_details functional agent...")
        invoice_input = InvoiceText(raw_text=sample_invoice_text)
        extracted_data: ExtractedInvoice = extract_invoice_details(invoice_input)

        print("\nSuccessfully Extracted Invoice Details:")
        print(extracted_data.model_dump_json(indent=2))

    except ValidationFailedError as ve:
        print(f"\nValidation Failed after retries for invoice extraction: {ve}")
        context = ve.last_error_context
        if context:
            print("Last error context details:")
            print(f"  Function: {context.func_name}")
            print(f"  Attempts: {context.current_attempt_number}/{context.max_retries + 1}")
            for i, attempt_log in enumerate(context.get_attempts_history()):
                print(f"    Attempt {i + 1} Log:")
                if attempt_log.get("llm_response"):
                    print(f"      LLM Generated Code:\n'''{attempt_log['llm_response']}'''")
                if attempt_log.get("error"):
                    print(f"      Error: {attempt_log['error'].get('message', 'N/A')}")

    except MaxRetriesExceededError as mre:
        print(f"\nMax Retries Exceeded for invoice extraction: {mre}")
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
        print(f"\nAn unexpected error occurred during invoice extraction: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # --- Example 3: Smart Text Rewriter ---
    print("\n\n--- Task: Smart Text Rewriter ---")
    text_to_refine = "This particular piece of text is rather long-winded and verbose, and it would definitely benefit from being made more concise and to the point, perhaps by removing redundant words."
    char_limit = 80
    print(f'Input Text ({len(text_to_refine)} chars): "{text_to_refine}"')
    print(f"Target character limit: {char_limit}")

    try:
        print("\nCalling smart_rewriter functional agent...")
        refine_input = RefineTextToLimitInput(original_text=text_to_refine, max_length=char_limit)
        refined_result: RefineTextToLimitOutput = smart_rewriter(data=refine_input)

        print("\nSuccessfully refined text:")
        print(f'  Original ({refined_result.original_length} chars): "{refined_result.original_text}"')
        print(f'  Refined ({refined_result.refined_length} chars): "{refined_result.refined_text}"')
        print(f"  Within {char_limit} char limit: {refined_result.is_within_limit}")
        if not refined_result.is_within_limit:
            print(f"  WARNING: Refined text is NOT within the {char_limit} character limit!")
        if (
            refined_result.refined_length >= refined_result.original_length
            and refined_result.original_length > char_limit
        ):
            print(
                f"  WARNING: Refined text was not made shorter than the original, though original exceeded the limit."
            )

    except ValidationFailedError as ve:
        print(f"\nValidation Failed after retries for text refinement: {ve}")
        context = ve.last_error_context
        if context:
            print("Last error context details:")
            print(f"  Function: {context.func_name}")
            print(f"  Attempts: {context.current_attempt_number}/{context.max_retries + 1}")
            for i, attempt_log in enumerate(context.get_attempts_history()):
                print(f"    Attempt {i + 1} Log:")
                if attempt_log.get("llm_response"):
                    print(f"      LLM Generated Code:\n'''{attempt_log['llm_response']}'''")
                if attempt_log.get("error"):
                    print(f"      Error: {attempt_log['error'].get('message', 'N/A')}")
    except MaxRetriesExceededError as mre:
        print(f"\nMax Retries Exceeded for text refinement: {mre}")
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
        print(f"\nAn unexpected error occurred during text refinement: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    # --- End of Example 3 ---

    print("\nDemo finished.")


if __name__ == "__main__":
    # To run from the project root (llm-functional-agents/):
    # python -m examples.run_demo
    main()  # Changed from asyncio.run(main())
