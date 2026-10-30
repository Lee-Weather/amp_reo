"""CPU fixtures shared by the two independent reference-guided task branches."""
from types import SimpleNamespace

import torch
from humanoid.amp.runner import AMPOnPolicyRunner
from test_exp110 import DHFakeEnv
from test_exp113 import configuration113, transfer_config
from test_exp3 import definitions


def configuration114(route):
    namespace = configuration113()
    definitions("humanoid/envs/x1/x1_amp_exp114_common_config.py", namespace)
    definitions("humanoid/envs/x1/x1_amp_exp"+route+"_config.py", namespace)
    return namespace


def cpu_runner(library, phase=False, source=False):
    config = transfer_config()
    env = DHFakeEnv(library)
    config["runner"]["require_policy_transfer"] = not source
    config["runner"]["warm_start_mode"] = "policy_append_phase" if phase else "policy_only"
    if phase:
        env.num_obs, env.num_privileged_obs = 6468, 101
        env.obs, env.critic = torch.zeros(4, 6468), torch.zeros(4, 101)
        env.cfg.env = SimpleNamespace(frame_stack=66, short_frame_stack=5, num_single_obs=98)
        config["algorithm"]["state_estimator_obs_index"] = 98
    env.checkpoint_training_metadata = lambda: {"curriculum": "source" if source else ("phase" if phase else "distance")}
    return AMPOnPolicyRunner(env, config)
