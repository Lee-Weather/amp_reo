"""exp3.6 runner: unchanged machinery, the reset distribution lives in the env."""
from .x1_exp35_runner import X1Exp35AMPOnPolicyRunner


class X1Exp36AMPOnPolicyRunner(X1Exp35AMPOnPolicyRunner):
    """No runner-side change."""