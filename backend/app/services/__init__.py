# Package init for app.services
# Inject a clean stub for llm_enhanced_policy_service when the original module is corrupted.

import sys

try:
    # Try to import the real module first
    from . import llm_enhanced_policy_service as _real_llm_module  # noqa: F401
except Exception:
    # Fallback: expose the stub as the real module
    from .llm_enhanced_policy_service_stub import (
        LLMEnhancedPolicyService as _LLMEnhancedPolicyService,
        llm_enhanced_policy_service as _llm_enhanced_policy_service,
    )
    import types

    stub_module = types.ModuleType("app.services.llm_enhanced_policy_service")
    setattr(stub_module, "LLMEnhancedPolicyService", _LLMEnhancedPolicyService)
    setattr(stub_module, "llm_enhanced_policy_service", _llm_enhanced_policy_service)

    # Register stub under the expected module path so `from app.services.llm_enhanced_policy_service import ...` works
    sys.modules["app.services.llm_enhanced_policy_service"] = stub_module

