"""Use the isolated publication workflow for the contact-domain experiment."""
import publish_exp210 as p

p.EXPERIMENT='exp2.11'
p.PARENT='0dfb6838bc5195a36652890f2ddb996b99c40da1'
p.BRANCH='codex/exp2-11'
p.FILES=['humanoid/amp/contact_domain.py','humanoid/amp/exp211_transfer.py',
    'humanoid/envs/x1/x1_amp_exp211_config.py','humanoid/envs/x1/x1_amp_exp211_env.py',
    'humanoid/scripts/train_contact_amp.py','tests/test_exp211.py','czy/exp1/exp2.md']
p.REGISTRATION="""
from .x1.x1_amp_exp211_config import X1AMPExp211Cfg,X1AMPExp211CfgPPO
from .x1.x1_amp_exp211_env import X1AMPExp211Env

task_registry.register('x1_amp_exp211',X1AMPExp211Env,X1AMPExp211Cfg(),X1AMPExp211CfgPPO())
"""
p.TEST_PATTERNS=['test_exp211.py']
p.ENTRIES=['humanoid/scripts/train_contact_amp.py','humanoid/scripts/resume_amp.py',
    'humanoid/amp/exp211_transfer.py','humanoid/amp/contact_domain.py',
    'humanoid/envs/x1/x1_amp_exp211_config.py','humanoid/envs/x1/x1_amp_exp211_env.py','humanoid/envs/__init__.py']
p.MESSAGE=b'Train exp2.11 contact-offset robustness from full13500 state; preserve rewards and controller\n'

if __name__=='__main__':p.main()
