import pytest
from unittest.mock import MagicMock, patch
import importlib.util
import os
import sys

# Load triage_core directly to avoid circular import via src.rag.__init__
_spec = importlib.util.spec_from_file_location(
    "triage_core",
    os.path.join(os.path.dirname(__file__), "..", "src", "rag", "triage_core.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["triage_core"] = _mod
_spec.loader.exec_module(_mod)
fetch_vertex_pricing = _mod.fetch_vertex_pricing

class MockSKU:
    def __init__(self, description, unit="count", price=0.0):
        self.description = description

        # Mock pricing expression
        class Rate:
            class UnitPrice:
                def __init__(self, units, nanos):
                    self.units = units
                    self.nanos = nanos

            def __init__(self, price):
                self.unit_price = self.UnitPrice(int(price), int((price - int(price)) * 1e9))

        class Expression:
            def __init__(self, unit, price):
                self.usage_unit = unit
                self.tiered_rates = [Rate(price)]

        class PricingInfo:
            def __init__(self, unit, price):
                self.pricing_expression = Expression(unit, price)

        self.pricing_info = [PricingInfo(unit, price)]


@patch("google.cloud.billing_v1.CloudCatalogClient")
def test_fetch_vertex_pricing_success(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    skus = [
        MockSKU("Gemini 2.5 Flash GA Video Input Priority - Predictions", "count", 0.0000005),
        MockSKU("Gemini 2.5 Flash Input Text Caching Storage", "h", 0.000001),
        MockSKU("Gemini 2.5 Flash Text Input - Predictions", "count", 0.00000015),
        MockSKU("Gemini 2.5 Flash Text Output - Predictions", "count", 0.0000045),
        MockSKU("Gemini 2.5 Flash Thinking Text Output - Predictions", "count", 0.000006),
        MockSKU("Gemini 2.5 Flash GA Text Input - Predictions", "count", 0.0000003),
        MockSKU("Gemini 2.5 Flash GA Text Output - Predictions", "count", 0.0000025),
        MockSKU("Gemini 2.5 Flash GA Thinking Text Output - Predictions", "count", 0.0000025),
        MockSKU("Gemini 2.5 Pro Text Input - Predictions", "count", 0.00000125),
        MockSKU("Gemini 2.5 Pro Text Output - Predictions", "count", 0.000015),
        MockSKU("Gemini 2.5 Pro Text Input Caching (Long)", "count", 0.00000025),
    ]

    mock_client.list_skus.return_value = skus

    # Test Flash — should prefer GA SKUs
    pricing, source = fetch_vertex_pricing("gemini-2.5-flash")
    assert source == "billing_catalog"
    assert pricing is not None
    assert pricing["input"] == pytest.approx(0.0000003)   # GA input
    assert pricing["output"] == pytest.approx(0.0000025)   # GA output
    assert pricing["thinking"] == pytest.approx(0.0000025)  # GA thinking

    # Test Pro
    pricing_pro, source_pro = fetch_vertex_pricing("gemini-2.5-pro")
    assert source_pro == "billing_catalog"
    assert pricing_pro is not None
    assert pricing_pro["input"] == pytest.approx(0.00000125)
    assert pricing_pro["output"] == pytest.approx(0.000015)


def test_fetch_vertex_pricing_fallback():
    with patch("google.cloud.billing_v1.CloudCatalogClient", side_effect=Exception("Failed to connect")):
        pricing, source = fetch_vertex_pricing("gemini-2.5-flash")
        assert source == "fallback_static"
        assert pricing is not None
        assert pricing["input"] == pytest.approx(0.30 / 1_000_000)
        assert pricing["output"] == pytest.approx(2.50 / 1_000_000)
        assert pricing["thinking"] == pytest.approx(2.50 / 1_000_000)

def test_fetch_vertex_pricing_unavailable():
    with patch("google.cloud.billing_v1.CloudCatalogClient", side_effect=Exception("Failed to connect")):
        pricing, source = fetch_vertex_pricing("unknown-model")
        assert source == "unavailable"
        assert pricing is None


def test_anthropic_models_have_fallback_pricing():
    """All supported Anthropic models must have fallback pricing entries."""
    FALLBACK_PRICING = _mod.FALLBACK_PRICING
    anthropic_models = [k for k in FALLBACK_PRICING if k.startswith("claude-")]
    assert len(anthropic_models) >= 3, f"Expected ≥3 Anthropic models, got {anthropic_models}"

    for model_id in anthropic_models:
        pricing = FALLBACK_PRICING[model_id]
        assert "input" in pricing, f"{model_id} missing 'input' key"
        assert "output" in pricing, f"{model_id} missing 'output' key"
        assert pricing["input"] > 0, f"{model_id} input pricing is zero"
        assert pricing["output"] > 0, f"{model_id} output pricing is zero"


def test_haiku_fallback_pricing_returns_nonzero():
    """Haiku pricing must be returned via fallback (not Billing Catalog)."""
    with patch("google.cloud.billing_v1.CloudCatalogClient", side_effect=Exception("no billing")):
        pricing, source = fetch_vertex_pricing("claude-haiku-4-5")
        assert source == "fallback_static"
        assert pricing is not None
        assert pricing["input"] == pytest.approx(1.00 / 1_000_000)
        assert pricing["output"] == pytest.approx(5.00 / 1_000_000)


def test_haiku_cost_computation_nonzero():
    """End-to-end: Haiku pricing × realistic token counts → non-zero cost.

    Regression test for postmortem finding #15: all Haiku runs showed cost=0
    because the fallback pricing entry didn't exist at the time.
    """
    from src.rag.agentic_pipeline import TriageAgenticPipeline, CostBudget
    from src.rag.query_agents import NoopQueryAgent

    with patch("google.cloud.billing_v1.CloudCatalogClient", side_effect=Exception("no billing")):
        pricing, _ = fetch_vertex_pricing("claude-haiku-4-5")

    pipeline = TriageAgenticPipeline(
        query_agent=NoopQueryAgent(),
        top_k=0,
        context_chars=8000,
        model_id="claude-haiku-4-5",
        pricing=pricing,
        cost_budget=CostBudget(),
        retrieval_cache={},
        run_id="test",
    )

    # Simulate token counts from a real Haiku fewshot run (150 rows)
    cost = pipeline._compute_cost(
        prompt_tokens=208950,
        completion_tokens=1650,
        thinking_tokens=0,
    )
    assert cost > 0, f"Haiku cost should be >0, got {cost}"
    assert cost == pytest.approx(0.2172, rel=0.01)


@patch("google.cloud.billing_v1.CloudCatalogClient")
def test_fetch_vertex_pricing_no_ga_falls_back_to_non_ga(mock_client_class):
    """When no GA SKU exists, non-GA should be used."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    skus = [
        MockSKU("Gemini 2.5 Flash Text Input - Predictions", "count", 0.00000015),
        MockSKU("Gemini 2.5 Flash Text Output - Predictions", "count", 0.0000045),
        MockSKU("Gemini 2.5 Flash Thinking Text Output - Predictions", "count", 0.000006),
    ]
    mock_client.list_skus.return_value = skus

    pricing, source = fetch_vertex_pricing("gemini-2.5-flash")
    assert source == "billing_catalog"
    assert pricing["input"] == pytest.approx(0.00000015)
    assert pricing["output"] == pytest.approx(0.0000045)
    assert pricing["thinking"] == pytest.approx(0.000006)
