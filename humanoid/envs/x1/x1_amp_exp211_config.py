"""exp2.11: cover contact anticipation differences, with unchanged source rewards."""
from .x1_amp_exp29_config import X1AMPExp29Cfg,X1AMPExp29CfgPPO


class X1AMPExp211Cfg(X1AMPExp29Cfg):
    class domain_rand(X1AMPExp29Cfg.domain_rand):
        randomize_contact_offset=True
        contact_offset_range=[.001,.020]


class X1AMPExp211CfgPPO(X1AMPExp29CfgPPO):
    class runner(X1AMPExp29CfgPPO.runner):
        experiment_name='x1_amp_exp211'
        run_name='exp2.11'
        max_iterations=1000
