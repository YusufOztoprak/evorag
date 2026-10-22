from decimal import Decimal

from evorag.observability.pricing import embedding_cost_usd


def test_cost_for_known_model():
    cost = embedding_cost_usd("text-embedding-3-small", 18)

    assert cost == Decimal("0.00000036")


def test_unknown_model_returns_none():
    cost = embedding_cost_usd("unknown-model", 1_000_000)

    assert cost is None

def test_cost_for_real_request():
    cost = embedding_cost_usd("text-embedding-3-small", 18)

    assert cost == Decimal("0.00000036")
