import pytest

from agentreclab.protocols.events import RevealGate


def test_reveal_observe_commit():
    gate = RevealGate()
    with pytest.raises(RuntimeError, match="revealed"):
        gate.observe()
    gate.reveal()
    gate.observe()
    gate.commit()
    with pytest.raises(RuntimeError, match="committed"):
        gate.observe()
