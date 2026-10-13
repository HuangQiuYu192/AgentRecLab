from models.i2agent import I2Agent
from models.iagent import IAgent
from models.memrec import MemRec


def create_model(name, use_batch=False):
    models = {"iagent": IAgent, "i2agent": I2Agent, "memrec": MemRec}
    if name not in models:
        raise ValueError(f"unsupported model: {name}")
    return models[name](use_batch=use_batch)
