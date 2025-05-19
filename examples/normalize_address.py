from typing import Optional

from pydantic import BaseModel, Field

from llm_functional_agents import llm_func


class RawAddress(BaseModel):
    address_string: str = Field(..., description="The raw, unstructured address string.")


class NormalizedAddress(BaseModel):
    street: Optional[str] = Field(None, description="Street name and number.")
    city: Optional[str] = Field(None, description="City name.")
    state: Optional[str] = Field(None, description="State abbreviation (e.g., CA, NY).")
    zip_code: Optional[str] = Field(None, description="5-digit ZIP code.")
    country: str = Field("US", description="Country code (defaults to US).")

    # Example of a simple post-condition assertion (more complex ones handled by agent_executor)
    def _validate_zip_code_format(cls, v):
        if v and not (isinstance(v, str) and len(v) == 5 and v.isdigit()):
            raise ValueError("ZIP code must be a 5-digit string.")
        return v

    # In the actual library, assertions would be defined differently, likely in the decorator or agent_executor


# ---- Assertion Hook Example ----
def assert_state_format(normalized_address_output: NormalizedAddress, raw_input: RawAddress):
    """Asserts that the state, if provided, is a 2-letter uppercase string."""
    if normalized_address_output.state is not None:
        assert (
            len(normalized_address_output.state) == 2 and normalized_address_output.state.isupper()
        ), f"State '{normalized_address_output.state}' must be a 2-letter uppercase abbreviation."


# ------------------------------


@llm_func(output_model=NormalizedAddress, post_hooks=[assert_state_format])  # Added post-hook
def normalize_address(raw_address: RawAddress) -> NormalizedAddress:
    """
    Takes a raw, unstructured address string and normalizes it into a structured
    format with distinct fields for street, city, state, and ZIP code.
    Assumes US addresses by default. The LLM should generate Python code
    that performs this normalization. For example, it might use string parsing,
    regular expressions, or a series of conditional checks.

    The generated Python code should assign its final structured result (an instance
    of NormalizedAddress or a dictionary that can be converted to it) to a variable
    named `llm_output`.
    """
    # This function body will be effectively ignored in the final library,
    # as the agent_executor will take over. For now, it's a placeholder.
    # The LLM will be prompted to provide the logic.

    # Example of what the LLM might be guided to produce (for testing):
    # if "123 Main St" in raw_address.address_string and "Anytown" in raw_address.address_string:
    #     llm_output = {
    #         "street": "123 Main St",
    #         "city": "Anytown",
    #         "state": "CA",
    #         "zip_code": "90210"
    #     }
    # else:
    #     llm_output = {} # Or raise an error if unable to parse

    # In a real scenario, the LLM provides this logic as executable code string.
    # The agent_executor runs it in the sandbox_executor.
    # The result ('llm_output') is then validated against NormalizedAddress.
    pass


if __name__ == "__main__":
    # This section is for direct testing of this file if needed,
    # but the main demo will be in run_demo.py

    # Placeholder for how it might be called if agent_executor was here
    print("This is an example functional agent. Run it via run_demo.py.")
    # test_address = RawAddress(address_string="123 Main St, Anytown, CA 90210")
    # try:
    #     normalized = normalize_address(test_address) # This call would trigger the agent
    #     print("Normalized Address (mock):")
    #     # print(normalized.model_dump_json(indent=2))
    # except Exception as e:
    #     print(f"Error: {e}")
