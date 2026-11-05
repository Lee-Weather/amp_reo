"""A fixed deployment-condition cohort; never alter reference or policy dimensions."""
import copy

COHORT=dict(fraction=.25,standing_joint_noise=0.,observation_noise=False,
    forward_only=True,friction=.8,contact_offset_m=.001,
    stand_buffer_seconds=1.5,transition_seconds=2.,episode_seconds=20.,
    other_environments='exp211_contact_randomization_with_original_reset_noise_and_commands')


def configure_nominal_probe(cfg):
    """Set constant shape properties through the base importer's setter path."""
    cfg.amp.standing_joint_noise=0.
    # The base importer writes friction only when this flag is enabled. A
    # degenerate interval gives constant0.8, rather than the URDF default.
    cfg.domain_rand.randomize_friction=True
    cfg.domain_rand.friction_range=[.8,.8]
    cfg.domain_rand.restitution_range=[0.,0.]
    cfg.domain_rand.randomize_contact_offset=False
    cfg.sim.physx.contact_offset=.001


def cohort_mask(env_ids,num_envs,fraction=.25):
    if not 0<fraction<1:raise ValueError('Cohort must leave both training populations')
    return env_ids<int(num_envs*fraction)


def training_metadata(parent):
    result=copy.deepcopy(parent)
    result.update(curriculum='nominal_startup_coverage_v1',
        initialization='25pct_exact_standing_forward_75pct_original_mixture',
        nominal_reset_entrypoint='exp22.reset_idx_hook_v1',
        nominal_startup_training=copy.deepcopy(COHORT))
    return result
