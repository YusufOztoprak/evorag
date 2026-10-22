from decimal import Decimal

# USD per 1M input tokens. Source: OpenAI model pricing pages, checked 2026-10-22.
EMBEDDING_PRICES_PER_MILLION: dict[str, Decimal] = {
    "text-embedding-3-small": Decimal("0.02"),
    "text-embedding-3-large": Decimal("0.13"),
}

def embedding_cost_usd(model: str, tokens: int) -> Decimal | None:
    price = EMBEDDING_PRICES_PER_MILLION.get(model)
    if price is None:
        return None
    return price * tokens / 1_000_000
