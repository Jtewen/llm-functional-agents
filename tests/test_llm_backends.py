import unittest
from unittest import mock
import os
from copy import deepcopy
import sys

# Required imports from the library
from llm_functional_agents.core.llm_backends import OpenAIBackend
from llm_functional_agents.config import configure, _config_store
from llm_functional_agents.exceptions import ConfigurationError


class TestOpenAIBackendClientInitialization(unittest.TestCase):
    _original_config_store_backup = None
    backend_id_counter = 0 # To ensure unique backend_ids for each test config

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
        # Minimal default, specific tests will add their backend configs
        _config_store.update({"default_llm_backend_id": "some_default_if_needed"})
        
        # Ensure each test uses a slightly different backend_id to avoid config clashes
        # if a test fails to clean up its specific config (though setUpClass/tearDownClass should handle it).
        TestOpenAIBackendClientInitialization.backend_id_counter += 1
        self.test_backend_id = f"openai_test_backend_{TestOpenAIBackendClientInitialization.backend_id_counter}"

    def tearDown(self):
        # _config_store is cleared in setUp, and fully restored by tearDownClass.
        # Individual test configs for self.test_backend_id are thus cleared.
        pass

    @mock.patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_client_constructor):
        mock_instance = mock_openai_client_constructor.return_value
        api_key = "dummy_key_123"
        configure(llm_backends={
            self.test_backend_id: {"type": "openai", "config": {"api_key": api_key}}
        })
        
        backend = OpenAIBackend(self.test_backend_id)
        
        mock_openai_client_constructor.assert_called_once_with(api_key=api_key, base_url=None, default_headers=None)
        self.assertEqual(backend.client, mock_instance)

    @mock.patch("openai.OpenAI")
    def test_init_with_api_key_and_base_url(self, mock_openai_client_constructor):
        api_key = "dummy_key_456"
        base_url = "http://localhost:7890/v1"
        configure(llm_backends={
            self.test_backend_id: {"type": "openai", "config": {"api_key": api_key, "base_url": base_url}}
        })

        OpenAIBackend(self.test_backend_id)
        mock_openai_client_constructor.assert_called_once_with(api_key=api_key, base_url=base_url, default_headers=None)

    def test_init_missing_api_key_openai(self):
        configure(llm_backends={
             self.test_backend_id: {"type": "openai", "config": {}} # No api_key
        })
        with self.assertRaises(ConfigurationError) as context:
            OpenAIBackend(self.test_backend_id)
        self.assertIn("API key for backend", str(context.exception))
        self.assertIn("OPENAI_API_KEY", str(context.exception)) # Check for correct env var hint

    def test_init_missing_api_key_openrouter(self):
        configure(llm_backends={
             self.test_backend_id: {"type": "openrouter", "config": {}} # No api_key
        })
        with self.assertRaises(ConfigurationError) as context:
            OpenAIBackend(self.test_backend_id)
        self.assertIn("API key for backend", str(context.exception))
        self.assertIn("OPENROUTER_API_KEY", str(context.exception))


    @mock.patch("openai.OpenAI")
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY_FOR_TEST": "env_key_123"})
    def test_init_api_key_from_env_var_openai(self, mock_openai_client_constructor):
        # The config system's _substitute_env_vars handles this before OpenAIBackend init
        configure(llm_backends={
            self.test_backend_id: {"type": "openai", "config": {"api_key": "${OPENAI_API_KEY_FOR_TEST}"}}
        })
        OpenAIBackend(self.test_backend_id)
        mock_openai_client_constructor.assert_called_once_with(api_key="env_key_123", base_url=None, default_headers=None)

    @mock.patch("openai.OpenAI")
    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY_FOR_TEST": "env_key_456"})
    def test_init_api_key_from_env_var_openrouter(self, mock_openai_client_constructor):
        configure(llm_backends={
            self.test_backend_id: {"type": "openrouter", "config": {"api_key": "${OPENROUTER_API_KEY_FOR_TEST}"}}
        })
        OpenAIBackend(self.test_backend_id)
        mock_openai_client_constructor.assert_called_once_with(
            api_key="env_key_456", 
            base_url=None, 
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Default OpenRouter header
                "X-Title": "LLM Functional Agents"      # Default OpenRouter header
            }
        )

    @mock.patch("openai.OpenAI")
    def test_init_openrouter_type_with_default_headers(self, mock_openai_client_constructor):
        api_key = "or_dummy_key"
        configure(llm_backends={
            self.test_backend_id: {"type": "openrouter", "config": {"api_key": api_key}}
        })
        OpenAIBackend(self.test_backend_id)
        expected_headers = {
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "LLM Functional Agents"
        }
        mock_openai_client_constructor.assert_called_once_with(api_key=api_key, base_url=None, default_headers=expected_headers)

    @mock.patch("openai.OpenAI")
    def test_init_openrouter_type_with_custom_headers(self, mock_openai_client_constructor):
        api_key = "or_custom_key"
        site_url = "https://my.app.dev"
        app_name = "My Custom App"
        configure(llm_backends={
            self.test_backend_id: {
                "type": "openrouter", 
                "config": {
                    "api_key": api_key,
                    "site_url_header": site_url,
                    "app_name_header": app_name
                }
            }
        })
        OpenAIBackend(self.test_backend_id)
        expected_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name
        }
        mock_openai_client_constructor.assert_called_once_with(api_key=api_key, base_url=None, default_headers=expected_headers)

    def test_import_error_for_openai_library(self):
        # Simulate openai library not being importable
        original_openai_module = sys.modules.get("openai")
        sys.modules["openai"] = None # Or some other object that would cause ImportError on attribute access
        
        try:
            configure(llm_backends={
                self.test_backend_id: {"type": "openai", "config": {"api_key": "any_key_will_do"}}
            })
            with self.assertRaises(ConfigurationError) as context:
                OpenAIBackend(self.test_backend_id)
            self.assertIn("OpenAI client library not found", str(context.exception))
        finally:
            # Restore the original module if it existed, or remove our placeholder
            if original_openai_module:
                sys.modules["openai"] = original_openai_module
            elif "openai" in sys.modules: # if we set it to None and it wasn't there before
                del sys.modules["openai"]


if __name__ == "__main__":
    unittest.main()
