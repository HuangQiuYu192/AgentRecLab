from agentreclab.providers.mock import MockProvider
from agentreclab.providers.openai_compatible import OpenAICompatible


def create(config):
    if config["name"] == "mock":
        return MockProvider(config)
    if config["name"] in ("openai", "azure", "qwen", "compatible"):
        return OpenAICompatible(config)
    raise ValueError(f"unknown provider {config['name']}")
