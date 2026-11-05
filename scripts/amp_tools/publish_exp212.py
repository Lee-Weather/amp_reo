"""Publish exact tested nominal-startup code while preserving the user's index."""
import publish_exp210 as p

p.EXPERIMENT='exp2.12'
p.PARENT='2d883f014ae9d3bf963591e934dc14ec49259581'
p.BRANCH='codex/exp2-12'
p.FILES=['humanoid/amp/nominal_startup.py','humanoid/amp/exp212_transfer.py',
    'humanoid/envs/x1/x1_amp_exp212_config.py','humanoid/envs/x1/x1_amp_exp212_env.py',
    'humanoid/envs/x1/x1_amp_exp21_env.py','humanoid/scripts/train_nominal_amp.py',
    'tests/test_exp212.py']
p.REGISTRATION="""
from .x1.x1_amp_exp212_config import X1AMPExp212Cfg,X1AMPExp212CfgPPO
from .x1.x1_amp_exp212_env import X1AMPExp212Env

task_registry.register('x1_amp_exp212',X1AMPExp212Env,X1AMPExp212Cfg(),X1AMPExp212CfgPPO())
"""
p.TEST_PATTERNS=['test_exp212.py']
p.ENTRIES=['humanoid/scripts/train_nominal_amp.py','humanoid/scripts/resume_amp.py',
    'humanoid/amp/exp212_transfer.py','humanoid/amp/nominal_startup.py',
    'humanoid/envs/x1/x1_amp_exp212_config.py','humanoid/envs/x1/x1_amp_exp212_env.py',
    'humanoid/envs/x1/x1_amp_exp21_env.py','humanoid/envs/__init__.py']
p.MESSAGE=b'Train exp2.12 nominal startup cohort with unchanged rewards and full13500 continuation\n'

if __name__=='__main__':p.main()
