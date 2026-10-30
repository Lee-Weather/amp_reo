"""Same exp1.15 objective; exactly transplant the exp1.9 MLP action function."""
from .x1_amp_exp115_config import X1AMPExp115Cfg, X1AMPExp115CfgPPO


class X1AMPExp115tCfg(X1AMPExp115Cfg):
    pass


class X1AMPExp115tCfgPPO(X1AMPExp115CfgPPO):
    class runner(X1AMPExp115CfgPPO.runner):
        experiment_name = 'x1_amp_exp115t'
        run_name = 'exp1.15t'
        teacher_sha256 = 'a558ff08e29c8840665353a5b30f24674562c21b6fdb6c0c47ec0f0cfcc9e451'
