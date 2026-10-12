from agentreclab.models.iagent import I2Agent, IAgent
from agentreclab.models.memrec import MemRec


def create_model(name):
    models = {"iagent": IAgent, "i2agent": I2Agent, "memrec": MemRec}
    if name not in models:
        raise ValueError(f"unsupported model: {name}")
    return models[name]()
