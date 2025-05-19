"""
Global configuration settings for the llm-functional-agents library.

This module can handle loading configurations from environment variables, 
config files, or direct programmatic settings.
"""

from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# --- Default Settings ---
DEFAULT_MAX_RETRIES = 3
DEFAULT_LLM_BACKEND_ID = "default_openrouter"
DEFAULT_RETRY_ON_VALIDATION_FAILURE = True
DEFAULT_RETRY_ON_SANDBOX_ERROR = True
DEFAULT_RETRY_ON_BACKEND_ERROR = False  # Often backend errors are not transient

DEFAULT_SANDBOX_CPU_TIME_LIMIT_SECONDS = 5
DEFAULT_SANDBOX_MEMORY_LIMIT_MB = 256 # In Megabytes for easier config
DEFAULT_SANDBOX_WALL_TIME_LIMIT_SECONDS = 10

# --- Configuration Storage (Simple dict for now) ---
# In a more complex app, this might use a proper config library (e.g., Pydantic-settings, Dynaconf)
_config_store: Dict[str, Any] = {
    "max_retries": DEFAULT_MAX_RETRIES,
    "default_llm_backend_id": DEFAULT_LLM_BACKEND_ID,
    "retry_policies": {
        "default": {
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_on_validation_failure": DEFAULT_RETRY_ON_VALIDATION_FAILURE,
            "retry_on_sandbox_error": DEFAULT_RETRY_ON_SANDBOX_ERROR,
            "retry_on_backend_error": DEFAULT_RETRY_ON_BACKEND_ERROR,
        }
    },
    "sandbox_limits": { # New section for sandbox limits
        "default": {
            "cpu_time_seconds": DEFAULT_SANDBOX_CPU_TIME_LIMIT_SECONDS,
            "memory_limit_mb": DEFAULT_SANDBOX_MEMORY_LIMIT_MB,
            "wall_time_seconds": DEFAULT_SANDBOX_WALL_TIME_LIMIT_SECONDS,
        }
    },
    "llm_backends": {
        "default_openai": {  # Example placeholder
            "type": "openai",
            "api_key": None,  # Loaded from env var OPENAI_API_KEY typically
            "default_model": "gpt-3.5-turbo",
            "base_url": None,  # For OpenAI-compatible APIs
        },
        "default_openrouter": {
            "type": "openrouter",  # Using 'openrouter' as type, can reuse openai client logic
            "api_key": None,  # Will be loaded from OPENROUTER_API_KEY
            "default_model": "google/gemini-flash-1.5",  # Changed to Gemini Flash
            "base_url": "https://openrouter.ai/api/v1",
            "site_url_header": "http://localhost:3000",  # Optional: for local testing, as recommended by OpenRouter
            "app_name_header": "LLM Functional Agents Demo",  # Optional
        }
        # Add other backends here, e.g., "anthropic_claude", "local_ollama"
    },
}

# --- Accessor Functions ---


def get_setting(key: str, default: Optional[Any] = None) -> Any:
    """Retrieve a global setting."""
    # Basic implementation, could add env var overrides, etc.
    return _config_store.get(key, default)


def configure(**kwargs: Any) -> None:
    """Programmatically update global configurations."""
    # Simple merge, could be more sophisticated
    for key, value in kwargs.items():
        if key in _config_store and isinstance(_config_store[key], dict) and isinstance(value, dict):
            # Special handling for "sandbox_limits" to allow direct override of the default policy
            if key == "sandbox_limits" and "default" in value:
                 _config_store[key]["default"].update(value["default"])
            else:
                _config_store[key].update(value) # Original logic for other dicts
        else:
            _config_store[key] = value


def get_retry_policy(policy_id: Optional[str] = "default") -> Dict[str, Any]:
    """Get a specific retry policy configuration."""
    policies = _config_store.get("retry_policies", {})
    return policies.get(policy_id, policies.get("default", {}))


def get_sandbox_limits(policy_id: Optional[str] = "default") -> Dict[str, Any]:
    """Get a specific sandbox limits policy configuration."""
    policies = _config_store.get("sandbox_limits", {})
    # Fallback to the default policy if the specified one doesn't exist
    # or if a specific limit isn't in the specified policy but is in default.
    default_limits = policies.get("default", {})
    specific_limits = policies.get(policy_id, {})
    # Merge specific over default
    merged_limits = default_limits.copy()
    merged_limits.update(specific_limits)
    return merged_limits


def get_llm_backend_config(backend_id: Optional[str]) -> Dict[str, Any]:
    """Get configuration for a specific LLM backend."""
    if not backend_id:
        backend_id = _config_store.get("default_llm_backend_id", DEFAULT_LLM_BACKEND_ID)

    backends = _config_store.get("llm_backends", {})
    config = backends.get(backend_id)
    if not config:
        from .exceptions import ConfigurationError

        raise ConfigurationError(f"LLM backend '{backend_id}' not configured.")

    # Example: Load API key from environment variable if not set directly
    # This logic would be specific to each backend type
    backend_type = config.get("type")
    if backend_type == "openai" and not config.get("api_key"):
        import os

        config["api_key"] = os.getenv("OPENAI_API_KEY")
        # Warning for OpenAI key moved to llm_backends.py to avoid circular import with exceptions

    elif backend_type == "openrouter" and not config.get("api_key"):
        import os

        config["api_key"] = os.getenv("OPENROUTER_API_KEY")

    return config


def get_default_llm_backend_id() -> str:
    return _config_store.get("default_llm_backend_id", DEFAULT_LLM_BACKEND_ID)


# Example of how to ensure API keys are loaded at module import or first use.
# This is a simple approach; more robust systems might use a dedicated setup function.
# _default_backend_id = get_default_llm_backend_id()
# if _default_backend_id:
#     try:
#         get_llm_backend_config(_default_backend_id) # Trigger potential API key load/check
#     except Exception as e:
#         print(f"Initial config load warning for default backend '{_default_backend_id}': {e}")
