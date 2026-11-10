"""exp3.3 runner: same anchor/hard-rejection as exp3.2, with standard standing pose."""
from .x1_exp32_runner import X1Exp32AMPOnPolicyRunner


class X1Exp33AMPOnPolicyRunner(X1Exp32AMPOnPolicyRunner):
    """Identical to exp3.2 runner – the standing-pose changes are in the env."""
