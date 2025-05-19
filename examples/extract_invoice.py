import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator

from llm_functional_agents import llm_func

# --- Input and Output Models ---


class InvoiceText(BaseModel):
    raw_text: str = Field(..., description="The full raw text content of the invoice.")


class InvoiceItem(BaseModel):
    description: Optional[str] = Field(None, description="Description of the line item.")
    quantity: Optional[float] = Field(None, description="Quantity of the item.")
    unit_price: Optional[float] = Field(None, description="Unit price of the item.")
    total_price: Optional[float] = Field(None, description="Total price for this line item (quantity * unit_price).")


class ExtractedInvoice(BaseModel):
    invoice_id: Optional[str] = Field(None, description="The invoice number or ID.")
    vendor_name: Optional[str] = Field(None, description="Name of the vendor/seller.")
    customer_name: Optional[str] = Field(None, description="Name of the customer/buyer.")
    issue_date: Optional[datetime.date] = Field(None, description="Date the invoice was issued (YYYY-MM-DD).")
    due_date: Optional[datetime.date] = Field(None, description="Date the payment is due (YYYY-MM-DD).")
    subtotal: Optional[float] = Field(None, description="The total amount before taxes and discounts.")
    tax_amount: Optional[float] = Field(None, description="The total amount of tax applied.")
    total_amount_due: Optional[float] = Field(None, description="The final total amount due on the invoice.")
    line_items: List[InvoiceItem] = Field(default_factory=list, description="List of line items on the invoice.")
    currency_symbol: Optional[str] = Field(None, description="Currency symbol found on the invoice (e.g., $, €).")

    # Pydantic can handle date parsing to datetime.date if the string format is good.
    # We can add custom validators if more complex parsing is needed.


# --- Assertion Hooks ---


def assert_total_amount_matches_items(
    extracted_invoice: ExtractedInvoice, invoice_input: InvoiceText  # Original input for context if needed
):
    """Asserts financial consistency: sum of line items matches subtotal, and subtotal + tax matches total_amount_due."""
    # Check 1: Sum of line item totals should match the subtotal
    if extracted_invoice.line_items and extracted_invoice.subtotal is not None:
        calculated_total_from_items = sum(
            item.total_price for item in extracted_invoice.line_items if item.total_price is not None
        )
        assert abs(calculated_total_from_items - extracted_invoice.subtotal) < 0.01, (
            f"Sum of line items ({calculated_total_from_items:.2f}) does not match "
            f"subtotal ({extracted_invoice.subtotal:.2f})."
        )

    # Check 2: Subtotal + Tax Amount should match Total Amount Due
    # This check is only performed if all three relevant fields are populated.
    if (
        extracted_invoice.subtotal is not None
        and extracted_invoice.tax_amount is not None
        and extracted_invoice.total_amount_due is not None
    ):
        calculated_total_due = extracted_invoice.subtotal + extracted_invoice.tax_amount
        assert abs(calculated_total_due - extracted_invoice.total_amount_due) < 0.01, (
            f"Subtotal ({extracted_invoice.subtotal:.2f}) + Tax ({extracted_invoice.tax_amount:.2f}) = {calculated_total_due:.2f}, "
            f"which does not match total_amount_due ({extracted_invoice.total_amount_due:.2f})."
        )
    elif (
        extracted_invoice.line_items
        and extracted_invoice.total_amount_due is not None
        and extracted_invoice.subtotal is None
    ):
        # Fallback: if subtotal is not extracted, but line items and total_amount_due are,
        # revert to the simpler check. This might be the case for invoices without explicit subtotal/tax.
        # This path might indicate the LLM couldn't find separate subtotal/tax and put line item sum into total_amount_due.
        # Or, it means the invoice indeed has no tax. The LLM should ideally set tax_amount to 0 or None.
        calculated_total_from_items = sum(
            item.total_price for item in extracted_invoice.line_items if item.total_price is not None
        )
        if extracted_invoice.tax_amount is None or extracted_invoice.tax_amount == 0:
            assert (
                abs(calculated_total_from_items - extracted_invoice.total_amount_due) < 0.01
            ), f"Sum of line items ({calculated_total_from_items:.2f}) does not match total_amount_due ({extracted_invoice.total_amount_due:.2f}) when no tax is specified."
        # If tax_amount is present but subtotal is not, the primary check (subtotal + tax = total) is preferred but cannot run.
        # This fallback becomes less reliable if tax_amount IS specified but subtotal is NOT.
        # We are assuming here that if tax_amount is filled, subtotal should also ideally be filled by the LLM.


def assert_dates_are_logical(extracted_invoice: ExtractedInvoice, invoice_input: InvoiceText):
    """Asserts that if both issue_date and due_date are present, issue_date is not after due_date."""
    if extracted_invoice.issue_date and extracted_invoice.due_date:
        assert (
            extracted_invoice.issue_date <= extracted_invoice.due_date
        ), f"Issue date ({extracted_invoice.issue_date}) cannot be after due date ({extracted_invoice.due_date})."


# --- LLM Functional Agent ---


@llm_func(output_model=ExtractedInvoice, post_hooks=[assert_total_amount_matches_items, assert_dates_are_logical])
def extract_invoice_details(invoice_content: InvoiceText) -> ExtractedInvoice:
    """
    Analyzes the provided raw invoice text and extracts structured information.
    This includes details like invoice ID, vendor and customer names, issue and due dates,
    a list of line items (each with description, quantity, unit price, total price),
    the subtotal (sum of line items), tax_amount, and the overall total_amount_due.
    Dates should be in YYYY-MM-DD format. Currency symbols should be captured.
    Numeric values (prices, quantities, subtotal, tax, total) should be parsed as floats.

    The LLM should generate Python code to perform this extraction.
    The generated Python code must assign its final result (an instance of ExtractedInvoice
    or a dictionary that can be converted to it) to a variable named `llm_output`.

    Key financial fields to extract:
    - `line_items`: Each item should have `description`, `quantity`, `unit_price`, `total_price`.
    - `subtotal`: This should be the sum of all `line_item.total_price`.
    - `tax_amount`: Extract any specified tax amount. If tax is given as a percentage, calculate the amount.
                   If no tax is mentioned, this can be 0 or None.
    - `total_amount_due`: The final amount. This should ideally be consistent with subtotal + tax_amount.

    If specific fields cannot be found, they should be omitted (set to None or default value if applicable, e.g., 0 for tax_amount if none specified).
    Pay attention to the relationships between financial fields as per the assertion hooks that will be run on your output.
    """
    # The LLM will be prompted to provide the logic for this function.
    # For example, it might use regex for dates and amounts, and text splitting/searching
    # for names and line items.
    pass


if __name__ == "__main__":
    # This section is for direct testing or to show how it might be called.
    # The main demo will be updated in run_demo.py.
    print("This is an example functional agent for invoice extraction.")
    print("To run a full demo, update and use examples/run_demo.py.")

    # Example dummy invoice text (replace with a real one for actual testing via run_demo.py)
    sample_invoice = """
    INVOICE
    Vendor: ACME Corp
    Invoice ID: INV-2024-001
    Date: 2024-07-29
    Due Date: 2024-08-28
    Customer: John Doe

    Items:
    1. Product A - 2 units @ $50.00 each, Total: $100.00
    2. Service B - 1 hour @ $75.00, Total: $75.00

    TOTAL DUE: $175.00
    """
    # invoice_input = InvoiceText(raw_text=sample_invoice)
    # try:
    #     # In a real scenario, this call would trigger the full agent execution
    #     # details = extract_invoice_details(invoice_input)
    #     # print("\nExtracted Details (Simulated/Placeholder):")
    #     # print(details.model_dump_json(indent=2))
    #     print("\nSimulated call to extract_invoice_details would go here.")
    #     print("The actual execution is handled by the agent framework when called from run_demo.py")
    # except Exception as e:
    #     print(f"Error: {e}")
