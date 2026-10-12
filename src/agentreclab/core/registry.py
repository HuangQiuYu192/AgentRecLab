from agentreclab.models.custom.intent_collab import IntentCollab
from agentreclab.models.i2agent.adapter import I2AgentAdapter
from agentreclab.models.iagent.adapter import IAgentAdapter
from agentreclab.models.mars.adapter import MarsAdapter


def create_model(name):
    if name == "memrec":
        from agentreclab.models.memrec.adapter import MemRecAdapter

        return MemRecAdapter()
    entries = {
        "iagent": IAgentAdapter,
        "i2agent": I2AgentAdapter,
        "mars": MarsAdapter,
        "custom": IntentCollab,
    }
    if name not in entries:
        raise ValueError(f"unknown model {name}")
    return entries[name]()
