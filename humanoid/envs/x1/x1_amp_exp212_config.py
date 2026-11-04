"""exp2.12: explicit deployment-condition startup coverage, unchanged rewards."""
from .x1_amp_exp211_config import X1AMPExp211Cfg,X1AMPExp211CfgPPO


class X1AMPExp212Cfg(X1AMPExp211Cfg):
    class amp(X1AMPExp211Cfg.amp):
        nominal_startup_fraction=.25


class X1AMPExp212CfgPPO(X1AMPExp211CfgPPO):
    class runner(X1AMPExp211CfgPPO.runner):
        experiment_name='x1_amp_exp212'
        run_name='exp2.12'
        max_iterations=1000
