"""exp3.5 runner: same anchor/hard-rejection machinery, changes live in the env."""
from .x1_exp34_runner import X1Exp34AMPOnPolicyRunner


class X1Exp35AMPOnPolicyRunner(X1Exp34AMPOnPolicyRunner):
    """No runner-side change."""