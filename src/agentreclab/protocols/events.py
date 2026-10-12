from dataclasses import dataclass


@dataclass
class RevealGate:
    """Future prequential interface; explicit reveal must precede observe."""

    revealed: bool = False
    committed: bool = False

    def reveal(self):
        if self.committed:
            raise RuntimeError("already committed")
        self.revealed = True

    def observe(self):
        if not self.revealed:
            raise RuntimeError("feedback has not been revealed")
        if self.committed:
            raise RuntimeError("feedback already committed")

    def commit(self):
        self.observe()
        self.committed = True
