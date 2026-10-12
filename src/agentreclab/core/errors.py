class ValidationError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class ProtocolMismatch(ValueError):
    pass
