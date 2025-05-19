"""
Abstracts interactions with various LLM providers.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..config import get_default_llm_backend_id, get_llm_backend_config
from ..exceptions import ConfigurationError, LLMBackendError


# --- LLMBackend Base Class ---
class LLMBackend(ABC):
    """Abstract Base Class for all LLM backends."""

    def __init__(self, backend_id: str):
        self.backend_id = backend_id
        self.config = get_llm_backend_config(backend_id) # Fetches merged config
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the specific LLM client (e.g., OpenAI client)."""
        pass

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Sends a prompt to the LLM and returns the raw string response.
        
        kwargs can include model-specific parameters like temperature, max_tokens, etc.,
        which should be passed through to the underlying client call if supported.
        """
        pass

    # Consider adding a `stream` method in the future for streaming responses.

# --- Concrete Backend Implementations ---

class OpenAIBackend(LLMBackend):
    """LLM Backend for OpenAI models and compatible APIs like OpenRouter."""

    def _initialize_client(self) -> Any:
        try:
            import openai
        except ImportError:
            raise ConfigurationError(
                "OpenAI client library not found. Please install with `pip install openai`."
            )
        
        api_key = self.config.get("api_key") # Already loaded by get_llm_backend_config
        if not api_key:
            # This check is important here, as get_llm_backend_config might not raise for a missing key
            # if it's just a warning there.
            env_var_name = "OPENAI_API_KEY" if self.config.get("type") == "openai" else "OPENROUTER_API_KEY"
            raise ConfigurationError(
                f"API key for backend '{self.backend_id}' (type: {self.config.get("type")}) not found. "
                f"Set via {env_var_name} environment variable or in config."
            )
        
        base_url = self.config.get("base_url")
        default_headers = None
        if self.config.get("type") == "openrouter":
            default_headers = {
                "HTTP-Referer": self.config.get("site_url_header", "http://localhost:3000"),
                "X-Title": self.config.get("app_name_header", "LLM Functional Agents")
            }

        return openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        model_name = self.config.get("default_model", "gpt-3.5-turbo")
        # Allow overriding model via invoke kwargs
        model_to_use = kwargs.pop("model", model_name) 

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                **kwargs # Pass through other OpenAI params like temperature, max_tokens
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMBackendError(f"OpenAI response content is None for backend '{self.backend_id}'.")
            return content
        except Exception as e:
            # Catch OpenAI specific errors if possible, e.g., openai.APIError
            raise LLMBackendError(f"OpenAI API error for backend '{self.backend_id}': {e}") from e

# Add other backends like AnthropicBackend, LocalModelBackend (e.g., Ollama) here...
# class AnthropicBackend(LLMBackend): ...
# class OllamaBackend(LLMBackend): ...

# --- Backend Factory and Cache ---
_cached_llm_backends: Dict[str, LLMBackend] = {}

def get_llm_backend(backend_id: Optional[str] = None) -> LLMBackend:
    """
    Factory function to get an initialized LLMBackend instance.
    Uses a cache to avoid re-initializing clients for the same backend_id.
    """
    if backend_id is None:
        backend_id = get_default_llm_backend_id()
    
    if not backend_id:
        raise ConfigurationError("No LLM backend ID specified and no default is configured.")

    if backend_id in _cached_llm_backends:
        return _cached_llm_backends[backend_id]

    backend_conf = get_llm_backend_config(backend_id) # This will raise ConfigurationError if not found
    backend_type = backend_conf.get("type")

    instance: LLMBackend
    if backend_type == "openai" or backend_type == "openrouter": # Modified to include openrouter
        instance = OpenAIBackend(backend_id=backend_id)
    # elif backend_type == "anthropic":
    #     instance = AnthropicBackend(backend_id=backend_id)
    # elif backend_type == "ollama":
    #     instance = OllamaBackend(backend_id=backend_id)
    else:
        raise ConfigurationError(
            f"Unsupported LLM backend type '{backend_type}' specified for backend_id '{backend_id}'."
        )
    
    _cached_llm_backends[backend_id] = instance
    return instance 