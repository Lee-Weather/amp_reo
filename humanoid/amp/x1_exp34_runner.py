"""exp3.4 runner: identical to exp3.3, the changes live in the env."""
from .x1_exp33_runner import X1Exp33AMPOnPolicyRunner


class X1Exp34AMPOnPolicyRunner(X1Exp33AMPOnPolicyRunner):
    """No runner-side change; keeps the exp3.2 anchor + hard-rejection machinery."""