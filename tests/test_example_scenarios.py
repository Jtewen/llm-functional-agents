import unittest
from unittest import mock
from copy import deepcopy
import datetime # For extract_invoice
from typing import Optional, List, Any # For Pydantic models and typing

from pydantic import BaseModel, Field, validator

from llm_functional_agents.core.llm_function import llm_func
from llm_functional_agents.core.llm_backends import LLMBackend, LLMBackendConfig
from llm_functional_agents.config import configure, _config_store, get_llm_backend_config
from llm_functional_agents.exceptions import ValidationFailedError, LLMOutputProcessingError


# --- Simplified Mock LLM Backend (adapted from test_integration.py) ---
class MockScenarioLLMBackend(LLMBackend):
    def __init__(self, backend_id: str):
        # Ensure this backend_id is configured before super().__init__
        self._original_config_for_id = None
        global _config_store
        if backend_id in _config_store.get("llm_backends", {}):
             self._original_config_for_id = deepcopy(_config_store["llm_backends"][backend_id])

        # Default mock config if none provided by test setup
        if "llm_backends" not in _config_store or backend_id not in _config_store["llm_backends"]:
            configure(llm_backends={backend_id: {"type": "mock_scenario_type", "config": {"model": "mock-scenario-model"}}})
        
        super().__init__(backend_id=backend_id)
        
        self.last_prompt_received: Optional[str] = None
        self.response_to_return: str = "llm_output = {}" # Default to empty dict
        self.call_count: int = 0

    def _initialize_client(self) -> Any:
        return "mock_scenario_client_instance"

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt_received = prompt
        self.call_count += 1
        return self.response_to_return
    
    def restore_config_for_id(self):
        global _config_store
        if self._original_config_for_id is not None:
            if "llm_backends" not in _config_store:
                _config_store["llm_backends"] = {}
            _config_store["llm_backends"][self.backend_id] = self._original_config_for_id
        elif self.backend_id in _config_store.get("llm_backends", {}):
            try:
                del _config_store["llm_backends"][self.backend_id]
                if not _config_store["llm_backends"]:
                    del _config_store["llm_backends"]
            except KeyError:
                pass

# --- Test Classes Setup ---
class BaseExampleScenarioTest(unittest.TestCase):
    _original_config_store_backup = None
    backend_id_counter = 0

    @classmethod
    def setUpClass(cls):
        global _config_store
        cls._original_config_store_backup = deepcopy(_config_store)

    @classmethod
    def tearDownClass(cls):
        global _config_store
        _config_store.clear()
        _config_store.update(cls._original_config_store_backup)

    def setUp(self):
        global _config_store
        _config_store.clear()
        
        BaseExampleScenarioTest.backend_id_counter += 1
        self.mock_backend_id = f"mock_example_backend_{BaseExampleScenarioTest.backend_id_counter}"
        
        # Configure the specific backend for this test instance
        configure(llm_backends={
            self.mock_backend_id: {"type": "mock_scenario_type", "config": {"model": "mock-model"}}
        })
        # Set it as default so @llm_func picks it up without explicit id, or pass it if decorator allows
        _config_store["default_llm_backend_id"] = self.mock_backend_id

        self.mock_backend_instance = MockScenarioLLMBackend(backend_id=self.mock_backend_id)
        
        self.patcher = mock.patch('llm_functional_agents.core.agent_executor.get_llm_backend')
        self.mock_get_llm_backend = self.patcher.start()
        self.mock_get_llm_backend.return_value = self.mock_backend_instance
    
    def tearDown(self):
        self.patcher.stop()
        if hasattr(self.mock_backend_instance, 'restore_config_for_id'):
            self.mock_backend_instance.restore_config_for_id()
        # Global config store restored by tearDownClass


# --- simple_categorizer.py Structures ---
class CategorizerTextInput(BaseModel):
    text: str = Field(..., description="The text to categorize.")

class CategorizerCategoryOutput(BaseModel):
    category: str = Field(..., description="The determined category for the text.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")

ALLOWED_CATEGORIES = ["general_inquiry", "technical_support", "billing_question", "positive_feedback"]

def sc_check_category(output: CategorizerCategoryOutput, *, data: CategorizerTextInput):
    assert output.category in ALLOWED_CATEGORIES, \
        f"Category '{output.category}' is not one of the allowed categories: {ALLOWED_CATEGORIES}"

@llm_func(output_model=CategorizerCategoryOutput, post_hooks=[sc_check_category])
def sc_categorize_text(data: CategorizerTextInput) -> CategorizerCategoryOutput:
    """
    Analyzes the input 'data.text' and determines its category.
    The category MUST be one of: general_inquiry, technical_support, billing_question, positive_feedback.
    """
    raise NotImplementedError("LLM should provide logic.")


# --- TestSimpleCategorizerScenario ---
class TestSimpleCategorizerScenario(BaseExampleScenarioTest):

    def test_success_categorization(self):
        input_text = "I love this app!"
        self.mock_backend_instance.response_to_return = 'llm_output = {"category": "positive_feedback", "confidence": 0.99}'
        
        result = sc_categorize_text(data=CategorizerTextInput(text=input_text))
        
        self.assertIsInstance(result, CategorizerCategoryOutput)
        self.assertEqual(result.category, "positive_feedback")
        self.assertEqual(result.confidence, 0.99)
        self.assertIn(input_text, self.mock_backend_instance.last_prompt_received)

    def test_hook_violation_unknown_category(self):
        input_text = "This is a test."
        self.mock_backend_instance.response_to_return = 'llm_output = {"category": "unknown_category", "confidence": 0.7}'
        
        with self.assertRaises(LLMOutputProcessingError) as context: # Wraps ValidationFailedError
            sc_categorize_text(data=CategorizerTextInput(text=input_text))
        
        # Check that the underlying error is indeed ValidationFailedError related to the hook
        self.assertIn("ValidationFailedError", str(context.exception.__cause__))
        self.assertIn("unknown_category", str(context.exception.__cause__))
        self.assertIn("not one of the allowed categories", str(context.exception.__cause__))

    def test_success_missing_confidence(self):
        input_text = "What are your hours?"
        self.mock_backend_instance.response_to_return = 'llm_output = {"category": "general_inquiry"}'
        
        result = sc_categorize_text(data=CategorizerTextInput(text=input_text))
        
        self.assertIsInstance(result, CategorizerCategoryOutput)
        self.assertEqual(result.category, "general_inquiry")
        self.assertIsNone(result.confidence)


# --- extract_invoice.py Structures ---
class EIInvoiceText(BaseModel):
    raw_text: str = Field(..., description="The full raw text content of the invoice.")

class EIInvoiceItem(BaseModel):
    description: Optional[str] = Field(None, description="Description of the line item.")
    quantity: Optional[float] = Field(None, description="Quantity of the item.")
    unit_price: Optional[float] = Field(None, description="Unit price of the item.")
    total_price: Optional[float] = Field(None, description="Total price for this line item (quantity * unit_price).")

class EIExtractedInvoice(BaseModel):
    invoice_id: Optional[str] = Field(None, description="The invoice number or ID.")
    vendor_name: Optional[str] = Field(None, description="Name of the vendor/seller.")
    customer_name: Optional[str] = Field(None, description="Name of the customer/buyer.")
    issue_date: Optional[datetime.date] = Field(None, description="Date the invoice was issued (YYYY-MM-DD).")
    due_date: Optional[datetime.date] = Field(None, description="Date the payment is due (YYYY-MM-DD).")
    subtotal: Optional[float] = Field(None, description="The total amount before taxes and discounts.")
    tax_amount: Optional[float] = Field(None, description="The total amount of tax applied.")
    total_amount_due: Optional[float] = Field(None, description="The final total amount due on the invoice.")
    line_items: List[EIInvoiceItem] = Field(default_factory=list, description="List of line items on the invoice.")
    currency_symbol: Optional[str] = Field(None, description="Currency symbol found on the invoice (e.g., $, €).")

def ei_assert_total_amount_matches_items(extracted_invoice: EIExtractedInvoice, invoice_input: EIInvoiceText):
    if extracted_invoice.line_items and extracted_invoice.subtotal is not None:
        calculated_total_from_items = sum(
            item.total_price for item in extracted_invoice.line_items if item.total_price is not None
        )
        assert abs(calculated_total_from_items - extracted_invoice.subtotal) < 0.01, \
            f"Sum of line items ({calculated_total_from_items:.2f}) does not match subtotal ({extracted_invoice.subtotal:.2f})."
    if (extracted_invoice.subtotal is not None and 
        extracted_invoice.tax_amount is not None and 
        extracted_invoice.total_amount_due is not None):
        calculated_total_due = extracted_invoice.subtotal + extracted_invoice.tax_amount
        assert abs(calculated_total_due - extracted_invoice.total_amount_due) < 0.01, \
            f"Subtotal + Tax ({calculated_total_due:.2f}) does not match total_amount_due ({extracted_invoice.total_amount_due:.2f})."

def ei_assert_dates_are_logical(extracted_invoice: EIExtractedInvoice, invoice_input: EIInvoiceText):
    if extracted_invoice.issue_date and extracted_invoice.due_date:
        assert extracted_invoice.issue_date <= extracted_invoice.due_date, \
            f"Issue date ({extracted_invoice.issue_date}) cannot be after due date ({extracted_invoice.due_date})."

@llm_func(output_model=EIExtractedInvoice, post_hooks=[ei_assert_total_amount_matches_items, ei_assert_dates_are_logical])
def ei_extract_invoice_details(invoice_content: EIInvoiceText) -> EIExtractedInvoice:
    """
    Analyzes raw invoice text and extracts structured information.
    """
    raise NotImplementedError("LLM should provide logic.")

# --- TestExtractInvoiceScenario ---
class TestExtractInvoiceScenario(BaseExampleScenarioTest):

    def test_success_simple_invoice(self):
        invoice_text = "Invoice ID: 123, Date: 2024-01-15, Total: $100, Item: Widget, Price: $100"
        mock_llm_output = {
            "invoice_id": "123", "vendor_name": "Test Vendor", "customer_name": "Test Customer",
            "issue_date": "2024-01-15", "due_date": "2024-02-15",
            "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 100.00, "total_price": 100.00}],
            "subtotal": 100.00, "tax_amount": 0.00, "total_amount_due": 100.00,
            "currency_symbol": "$"
        }
        self.mock_backend_instance.response_to_return = f"llm_output = {mock_llm_output!r}"
        
        result = ei_extract_invoice_details(invoice_content=EIInvoiceText(raw_text=invoice_text))
        
        self.assertIsInstance(result, EIExtractedInvoice)
        self.assertEqual(result.invoice_id, "123")
        self.assertEqual(result.total_amount_due, 100.00)
        self.assertEqual(result.issue_date, datetime.date(2024, 1, 15))
        self.assertIn(invoice_text, self.mock_backend_instance.last_prompt_received)

    def test_hook_violation_illogical_dates(self):
        invoice_text = "Date: 2024-03-15, Due: 2024-03-01" # Illogical
        mock_llm_output = {
            "issue_date": "2024-03-15", "due_date": "2024-03-01", # Illogical
            "total_amount_due": 50.00 # Other fields to make it seem valid otherwise
        }
        self.mock_backend_instance.response_to_return = f"llm_output = {mock_llm_output!r}"

        with self.assertRaises(LLMOutputProcessingError) as context:
            ei_extract_invoice_details(invoice_content=EIInvoiceText(raw_text=invoice_text))
        
        self.assertIn("ValidationFailedError", str(context.exception.__cause__))
        self.assertIn("cannot be after due date", str(context.exception.__cause__))

    def test_hook_violation_financial_mismatch(self):
        invoice_text = "Subtotal: 100, Tax: 10, Total: 105" # Mismatch
        mock_llm_output = {
            "subtotal": 100.00, "tax_amount": 10.00, "total_amount_due": 105.00, # Mismatch
            "issue_date": "2024-01-01", "due_date": "2024-01-31" # Logical dates
        }
        self.mock_backend_instance.response_to_return = f"llm_output = {mock_llm_output!r}"
        
        with self.assertRaises(LLMOutputProcessingError) as context:
            ei_extract_invoice_details(invoice_content=EIInvoiceText(raw_text=invoice_text))

        self.assertIn("ValidationFailedError", str(context.exception.__cause__))
        self.assertIn("does not match total_amount_due", str(context.exception.__cause__))


if __name__ == "__main__":
    unittest.main()
