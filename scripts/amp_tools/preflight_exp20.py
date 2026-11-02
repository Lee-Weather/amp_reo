"""Real model CPU preflight; software success is separate from reference acceptance."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'tests')]
from test_exp20 import configuration20,reward_environment
from preflight_reference_transfer import dictionary
from preflight_phase_refinement import environment
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.exp20_rewards import similarity
from humanoid.amp.stationary import CommandMotionSampler
from humanoid.amp.exp20_rewards import raw_violation
from humanoid.amp.motion import quat_rotate,quat_mul,quat_inverse_rotate


def reward_counterexamples(lib,reference,standing):
    cases = ['reference','standing','flat_moving','tiny_motion','locked_knees','waist_twist',
             'upper_deviation','wrong_phase','double_frequency','airborne']
    count = 126
    phases = (torch.arange(count)+.5)/count
    target = reference.state(phases)
    states = target.repeat(len(cases),1)
    raw = states[:,13:42].clone()
    n = len(lib.dof_names)
    def section(label):
        i=cases.index(label)
        return slice(i*count,(i+1)*count)
    for label in ('standing','flat_moving'):
        sl = section(label)
        raw[sl] = standing[13:42]
        states[sl,42:] = 0
        if label=='standing':
            states[sl,7:13] = 0
            states[sl,3:7] = standing[3:7]
    sl = section('tiny_motion')
    raw[sl] = standing[13:42]+.05*(raw[sl]-standing[13:42])
    states[sl,42:] *= .05
    knees = [i for i,name in enumerate(lib.dof_names) if 'knee_pitch' in name]
    sl = section('locked_knees')
    raw[sl,knees] = standing[13:42][knees]
    states[sl,42:][:,knees] = 0
    for label,selector,offset in (('waist_twist',lambda name:name.startswith('lumbar_'),.25),
                                ('upper_deviation',lambda name:'shoulder' in name,.6)):
        ids = [i for i,name in enumerate(lib.dof_names) if selector(name)]
        raw[section(label),ids] += offset
    for label,phase,factor in (('wrong_phase',phases+.5,1.),('double_frequency',2*phases,2.)):
        sl = section(label)
        wrong = reference.state(phase)
        raw[sl] = wrong[:,13:42]
        states[sl,42:] = factor*wrong[:,42:]
    states[:,13:42] = raw.maximum(lib.fk.lower).minimum(lib.fk.upper)
    states[:,2] += .002-reference.sole_height(states).amin(-1)
    states[section('airborne'),2] += .2
    env = reward_environment(lib,reference,len(states))
    env.gait_phase = phases.repeat(len(cases))
    env.standing_state = standing
    env.default_dof_pos = standing[13:42][None]
    env.root_states = states[:,:13]
    env.base_quat = env.root_states[:,3:7]
    env.dof_pos,env.dof_vel = states[:,13:42],states[:,42:]
    env.base_lin_vel = quat_inverse_rotate(env.base_quat,states[:,7:10])
    env.base_ang_vel = quat_inverse_rotate(env.base_quat,states[:,10:13])
    gravity = states.new_tensor([0.,0.,-1.]).expand(len(states),-1)
    env.projected_gravity = quat_inverse_rotate(env.base_quat,gravity)
    local,rot = lib.fk.key_transforms(env.dof_pos)
    env.rigid_state[:,:,:3] = states[:,None,:3]+quat_rotate(env.base_quat[:,None].expand(-1,6,-1),local)
    env.rigid_state[:,:,3:7] = quat_mul(env.base_quat[:,None].expand(-1,6,-1),rot)
    height = reference.sole_height(states)
    contact = (height<.008).float()
    contact[section('airborne')] = 0
    load = contact/contact.sum(-1,keepdim=True).clamp(min=1)
    env.contact_forces.zero_()
    env.contact_forces[:,:2,2] = load*36.46*9.81
    env.raw_violation_sum = raw_violation(raw,lib.fk.lower,lib.fk.upper)*env.cfg.control.decimation
    # Use consistent neighboring dq for acceleration, not an arbitrary zero prior.
    env.last_dof_vel = env.dof_vel.reshape(len(cases),count,n).roll(1,1).reshape(-1,n)
    env.compute_reward()
    report = {}
    for label in cases:
        sl = section(label)
        report[label] = dict(task_mean=float(env.rew_buf[sl].mean()),
            per_term={name:float(env.episode_sums[name][sl].mean()) for name in env.reward_names})
    full = report['reference']['task_mean']
    for label in cases[1:]:
        assert full>report[label]['task_mean'],(label,full,report[label])
    return dict(cases=report,passed=True,
        scope='Actual full reward dispatch on synthetic grounded snapshots, deterministic contact loads and zero torques/action-rate; no physics or cumulative-return proof.')


def check(source_path,correction_path=None):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    source = torch.load(source_path,map_location='cpu',weights_only=False)
    ns = configuration20()
    settings = ns['X1AMPExp20Cfg']()
    if correction_path is None and settings.amp.native_correction_file:
        correction_path = Path(settings.amp.native_correction_file.format(LEGGED_GYM_ROOT_DIR=ROOT))
    config = dictionary(ns['X1AMPExp20CfgPPO']())
    config.update(dictionary(settings))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==config['runner']['policy_transfer_source_sha256']
    lib = MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized',
        ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf',
        source['metadata']['dof_names'],settings.amp.key_body_names,
        dt=.02,history_steps=3,weights=settings.amp.motion_weights,ground_clearance=.002)
    correction = json.loads(correction_path.read_text()) if correction_path else None
    reference = NativeCycle(lib,contact_correction=correction)
    sampler = CommandMotionSampler(lib)
    reference.quiet_histories = lib.history(sampler.ids[sampler.quiet],sampler.times[sampler.quiet])
    report = reference.audit()
    transitions = reference.audit_transitions(torch.tensor(source['metadata']['control']['standing_state']))
    counterexamples = reward_counterexamples(lib,reference,torch.tensor(source['metadata']['control']['standing_state']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original = NativeCycle(lib)
    phases = torch.arange(630)/630
    a,b = original.state(phases),reference.state(phases)
    fig,axes = plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for i,side in enumerate(('left','right')):
        for row,joint in enumerate(('hip_pitch','knee_pitch')):
            index = lib.dof_names.index(side+'_'+joint+'_joint')
            axes[row,i].plot(phases.numpy(),a[:,13+index].rad2deg().numpy(),label='Closed source + limit margin')
            axes[row,i].plot(phases.numpy(),b[:,13+index].rad2deg().numpy(),label='Contact corrected')
            axes[row,i].set(title=side+' '+joint,ylabel='degrees')
            axes[row,i].legend()
        axes[2,i].plot(phases.numpy(),original.sole_height(a)[:,i].numpy()*1000,label='Before')
        axes[2,i].plot(phases.numpy(),reference.sole_height(b)[:,i].numpy()*1000,label='After')
        axes[2,i].axhline(10,color='gray',linestyle='--')
        axes[2,i].set(title=side+' collision sole clearance',ylabel='mm',xlabel='cycle phase')
        axes[2,i].legend()
    fig.suptitle('exp2.0 native1.26s reference: CPU kinematics, not physics acceptance')
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer,format='png',dpi=140)
    plt.close(fig)
    def make():
        env = environment(lib,settings,source['metadata'])
        env.checkpoint_training_metadata = lambda: dict(curriculum='native_full_body_v1',native_reference=report,
            reference_source_sha256={name:hashlib.sha256((ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'/(name+'.pkl')).read_bytes()).hexdigest() for name in lib.names})
        env.sample_amp_expert_by_category = reference.expert
        env.source_step = env.step
        def step(actions):
            obs,critic,reward,done,info = env.source_step(actions)
            labels = torch.tensor([True,False,True,True])
            info['amp_valid'] = torch.tensor([True,True,False,True])
            info['amp_moving'] = labels
            info['amp_obs'] = reference.expert(labels)+.001*torch.randn(4,3,86)
            info['amp_style_multiplier'] = torch.tensor([.5,.1,0.,.5])
            return obs,critic,reward,done,info
        env.step = step
        return env
    runner = AMPOnPolicyRunner(make(),copy.deepcopy(config))
    fresh_value = copy.deepcopy(runner.actor_critic.critic.state_dict())
    fresh_disc = copy.deepcopy(runner.discriminator.state_dict())
    with contextlib.redirect_stdout(io.StringIO()): runner.warm_start(str(source_path))
    assert runner.current_learning_iteration==0 and runner.replay.size==0
    assert not runner.optimizer.state and not runner.disc_optimizer.state
    diagnostic_config = copy.deepcopy(config)
    diagnostic_config['env'].update(num_envs=1,capture_replay_state=True,enable_camera_sensors=True,camera_width=1920,camera_height=1080)
    diagnostic_config['noise']['add_noise'] = False
    diagnostic_config['runner']['policy_transfer_diagnostic_mode'] = True
    baseline = AMPOnPolicyRunner(make(),diagnostic_config)
    with contextlib.redirect_stdout(io.StringIO()): baseline.warm_start(str(source_path))
    assert baseline.current_learning_iteration==0 and not baseline.optimizer.state
    del baseline
    for key,value in source['model_state_dict'].items():
        if not key.startswith('critic.'):
            assert torch.equal(value,runner.actor_critic.state_dict()[key]),key
    for key,value in fresh_value.items(): assert torch.equal(value,runner.actor_critic.critic.state_dict()[key])
    for key,value in fresh_disc.items(): assert torch.equal(value,runner.discriminator.state_dict()[key])
    runner.current_learning_iteration = 199
    assert runner._policy_learning_rate_cap()==1e-5
    runner.current_learning_iteration = 200
    assert runner._policy_learning_rate_cap()==3e-5
    runner.current_learning_iteration = 0
    # Match the production single-frame convention, including explicit phase.
    obs = torch.zeros(64,6468)
    obs.view(64,66,98)[:,:,0] = settings.commands.native_speed*2
    obs.view(64,66,98)[:,:,8] = -1
    obs.view(64,66,98)[:,:,97] = 1
    with torch.no_grad():
        for _ in range(64):
            output = runner.actor_critic.act_inference(obs)
            assert output.shape==(64,29) and torch.isfinite(output).all()
    batch = runner.collect_rollout()
    assert int(batch['amp_valid'].sum())==72
    policy_metrics = runner.update_policy(batch)
    disc_metrics = runner.update_discriminator(batch['amp'],batch['amp_valid'],batch['amp_moving'])
    assert runner.replay.size==72 and runner.optimizer.state and runner.disc_optimizer.state
    assert all(torch.isfinite(torch.tensor(v)) for v in {**policy_metrics,**disc_metrics}.values())
    runner.current_learning_iteration = 1
    with tempfile.TemporaryDirectory(prefix='exp20_preflight_') as directory:
        path = Path(directory)/'model_1.pt'
        with contextlib.redirect_stdout(io.StringIO()): runner.save(str(path))
        loaded = AMPOnPolicyRunner(make(),copy.deepcopy(config))
        loaded.load(str(path))
        assert loaded.current_learning_iteration==1 and loaded.training_origin==runner.training_origin
        for key,value in runner.actor_critic.state_dict().items(): assert torch.equal(value,loaded.actor_critic.state_dict()[key])
        module = torch.jit.script(copy.deepcopy(runner.actor_critic.get_policy_module()).cpu())
        torch.jit.save(module,str(Path(directory)/'policy.pt'))
        torch.testing.assert_close(module(obs),runner.actor_critic.act_inference(obs))
    phases = torch.arange(256)/256
    state = reference.state(phases)
    wrong = reference.state(phases+.5)
    groups = [('pose',reference.pitch_indices,.15,1.5),
        ('lateral',[i for i,n in enumerate(lib.dof_names) if any(j in n for j in ('hip_roll','hip_yaw','ankle_roll'))],.12,.75),
        ('waist',[i for i,n in enumerate(lib.dof_names) if n.startswith('lumbar_')],.12,1.),
        ('upper',[i for i,n in enumerate(lib.dof_names) if any(j in n for j in ('shoulder','elbow','wrist'))],.25,.75)]
    scores = {}
    for label,positions in [('native',state[:,13:42]),('wrong_phase',wrong[:,13:42]),
        ('standing',torch.tensor(source['metadata']['control']['standing_state'][13:42]).expand(256,-1))]:
        scores[label] = {name:float(similarity(positions[:,ids],state[:,13:42][:,ids],scale).mean()) for name,ids,scale,weight in groups}
        scores[label]['weighted_q'] = sum(scores[label][name]*weight for name,ids,scale,weight in groups)
    assert scores['native']['weighted_q']>scores['wrong_phase']['weighted_q']+.1
    assert scores['native']['weighted_q']>scores['standing']['weighted_q']+.1
    # Execute the actual train() entry, replacing only registry/learn to check
    # forwarding without claiming that3000 updates ran in this CPU preflight.
    entry_calls = []
    with tempfile.TemporaryDirectory(prefix='exp20_entry_') as directory:
        staged = Path(directory)/'input_model_7000.pt'
        staged.write_bytes(source_path.read_bytes())
        class Registry:
            def make_env(self,**kwargs):
                assert kwargs['name']=='x1_amp_exp20'
                return make(),settings
            def make_alg_runner(self,env,**kwargs):
                target = AMPOnPolicyRunner(env,copy.deepcopy(config))
                target.warm_start(str(staged))
                def learn(num_learning_iterations,init_at_random_ep_len):
                    assert target.current_learning_iteration==0
                    assert num_learning_iterations==3000 and not init_at_random_ep_len
                    entry_calls.append(dict(new_updates=num_learning_iterations,source_iteration=target.training_origin['source_iteration']))
                target.learn = learn
                return target,ns['X1AMPExp20CfgPPO'](),None
        tree = ast.parse((ROOT/'humanoid/scripts/train.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='train']
        namespace = dict(task_registry=Registry())
        exec(compile(tree,'production_train_entry','exec'),namespace)
        from types import SimpleNamespace
        args = SimpleNamespace(task='x1_amp_exp20',warm_start=str(staged),warm_start_sha256=config['runner']['policy_transfer_source_sha256'],warm_start_url_b64=None)
        with contextlib.redirect_stdout(io.StringIO()): namespace['train'](args)
    return dict(software_checks_passed=True,seed=5,source_sha256=config['runner']['policy_transfer_source_sha256'],
        inference_samples=4096,inference_dimensions=[6468,29],source_iteration=7000,new_iteration=0,
        retained=['actor','long_history','state_estimator','log_std'],
        reset=['critic','discriminator','normalizer','optimizers','replay'],
        policy_update_metrics=policy_metrics,discriminator_update_metrics=disc_metrics,reward_ordering=scores,
        full_reward_counterexamples=counterexamples,
        diagnostic_source_policy_transfer=True,production_train_entry=entry_calls,
        cpu_update_environment_count=4,
        reference_acceptance=report,transition_acceptance=transitions,
        ready_for_cloud=report['passed'] and transitions['passed'],
        _reference_plot_png=image_buffer.getvalue(),
        limitations=['CPU fixture executes production networks/PPO/AMP and numerical reference checks, not Isaac Gym physics.',
            'Full physical gait acceptance requires cloud simulation and recorded replay.',
            'Do not start long training while reference acceptance fails.'])


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference-correction',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = check(args.source,args.reference_correction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    plot = result.pop('_reference_plot_png')
    (args.output.parent/'reference_preflight.png').write_bytes(plot)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
