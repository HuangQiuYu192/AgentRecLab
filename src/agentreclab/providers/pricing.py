from datetime import date


def price(config, usage, input_estimate=0, output_estimate=0):
    if config["name"] == "mock":
        return 0.0, True
    prices = config.get("pricing")
    if not prices:
        return None, True
    date.fromisoformat(prices["effective_date"])
    if prices.get("currency") != "CNY":
        raise ValueError("pricing must be CNY; explicitly supply dated FX-converted prices")
    incoming = usage.input_tokens if usage else input_estimate
    outgoing = usage.output_tokens if usage else output_estimate
    cached = (usage.cached_input_tokens or 0) if usage else 0
    return (
        (incoming - cached) * prices["input_per_million"]
        + cached * prices.get("cached_per_million", prices["input_per_million"])
        + outgoing * prices["output_per_million"]
    ) / 1e6, usage is None
