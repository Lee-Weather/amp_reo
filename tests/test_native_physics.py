"""Regression: link-space inertia must not become a joint-space armature."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/amp_tools'))
from native_physics import align_runtime, merged_properties
from probe_dynamics import make_spec
from compare_dynamics_probes import compare


class NativePhysicsTest(unittest.TestCase):
    def test_physical_comparison_rejects_unmatched_inputs_and_detects_divergence(self):
        meta=dict(schema='x1_dynamics_probe_v1',case='air_replay',physics_dt=.002,
            checkpoint_sha256='abc',dof_names=['j'],no_contact_required=True,engine='fixture')
        roots=np.tile(np.array([0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],dtype=np.float32),(2,1))
        values=dict(initial_root=roots[0],initial_q=np.zeros(1),initial_dq=np.zeros(1),
            actions=np.zeros((2,1)),targets=np.zeros((2,1)),time_s=np.array([.002,.004]),
            q=np.zeros((2,1)),dq=np.zeros((2,1)),torques=np.zeros((2,1)),roots=roots,
            all_contact_force_max=np.zeros(2),metadata=np.array(json.dumps(meta)))
        with tempfile.TemporaryDirectory() as directory:
            left,right=Path(directory)/'left.npz',Path(directory)/'right.npz'
            np.savez(left,**values)
            changed={key:value.copy() for key,value in values.items()}
            changed['q'][1,0]=.06
            changed['roots'][:,3:7]*=-1  # Same orientation, opposite quaternion sign.
            np.savez(right,**changed)
            report=compare(left,right)
            self.assertEqual(report['first_joint_threshold_exceeded_s']['0.05'],.004)
            self.assertAlmostEqual(report['joint_error_max_rad'],.06)
            self.assertEqual(report['root_orientation_error_max_rad'],0.)
            self.assertTrue(report['valid_no_contact_comparison'])
            changed['all_contact_force_max'][0]=1.
            np.savez(right,**changed)
            self.assertFalse(compare(left,right)['valid_no_contact_comparison'])
            changed['actions'][0,0]=.1
            np.savez(right,**changed)
            with self.assertRaisesRegex(ValueError,'Unmatched input'):
                compare(left,right)

    def test_probe_cases_preserve_input_actions_and_isolate_ground_contact(self):
        trace = dict(initial_root=np.array([0.,0.,.6,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]),
            initial_q=np.array([.2]), initial_dq=np.zeros(1), initial_observation=np.zeros(3),
            observations=np.zeros((2,3)), actions=np.array([[1.],[2.]],dtype=np.float32),
            roots=np.array([[0.,0.,.6,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]]*2), q=np.zeros((2,1)),
            dq=np.zeros((2,1)), forces=np.zeros((2,2,3)), done=np.zeros(2), time_s=np.array([.02,.04]),
            metadata=np.array(json.dumps(dict(schema='x1_transition_trace_v1',checkpoint_sha256='abc',
                dof_names=['j'],num_observations=3,policy_dt=.02))))
        ck=dict(config=dict(sim=dict(dt=.002),control=dict(decimation=10,action_scale=.25),
            normalization=dict(clip_actions=5.),init_state=dict(default_joint_angles={'j':0.})),
            metadata=dict(dof_names=['j'],policy_dt=.02,num_obs=3,control=dict(hard_limits=[[-1.,1.]])))
        air=make_spec(trace,ck,'abc','air_replay',.04)
        ground=make_spec(trace,ck,'abc','ground_replay',.04)
        hold=make_spec(trace,ck,'abc','ground_hold',.04)
        np.testing.assert_array_equal(air['targets'],ground['targets'])
        np.testing.assert_allclose(air['root'][2]-ground['root'][2],1.5)
        np.testing.assert_allclose(ground['targets'][:10],.25)
        np.testing.assert_allclose(ground['targets'][10:],.5)
        np.testing.assert_allclose(hold['targets'],.2,atol=1e-7)
        fine=make_spec(trace,ck,'abc','air_replay',.04,physics_dt=.001)
        np.testing.assert_allclose(fine['targets'][:20],.25)
        np.testing.assert_allclose(fine['targets'][20:],.5)
        with self.assertRaisesRegex(ValueError,'periods disagree'):
            make_spec(trace,ck,'abc','air_replay',.04,physics_dt=.003)
        with self.assertRaisesRegex(ValueError,'bounded'):
            make_spec(trace,ck,'abc','air_replay',10.)

    def fixture(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="base"><freejoint/><inertial mass="2" pos="0 0 0" diaginertia=".2 .3 .4"/>
            <body name="hinge" pos="0 0 .4"><joint name="j" type="hinge"/>
              <inertial mass="1" pos=".1 0 0" diaginertia=".03 .04 .05"/>
              <body name="fixed" pos=".2 0 0"><inertial mass=".5" pos="0 0 0" diaginertia=".01 .02 .03"/></body>
            </body></body></worldbody></mujoco>''')
        grouped=merged_properties(model,['base','hinge'])
        bodies=[]
        for name,p in grouped.items():
            bodies.append(dict(name=name,mass=p['mass'],com=p['com'],inertia=(np.asarray(p['inertia'])+.01*np.eye(3)).tolist()))
        control=dict(stiffness=[10.],damping=[2.],torque_limits=[5.])
        cfg=dict(asset=dict(armature=.01),sim=dict(dt=.002))
        ck=dict(config=cfg,metadata=dict(control=control,dof_names=['j']))
        runtime=dict(asset_config=cfg['asset'],sim_config=cfg['sim'],final_bodies=bodies,
            checkpoint_sha256='fixture',dofs=dict(j=dict(armature=0.,damping=1.,friction=0.,driveMode=3,stiffness=0.)),**control)
        probe=dict(results=[dict(maximum_contact_force=0.)],repeat_max_state_delta=0.,damping_effect_max_joint_velocity_delta=.2)
        return model,ck,runtime,probe,grouped

    def test_full_tensor_and_fixed_group_count(self):
        m,ck,runtime,probe,before=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'r.json').write_text(json.dumps(runtime));(p/'p.json').write_text(json.dumps(probe))
            report=align_runtime(m,ck,p/'r.json',p/'p.json')
        after=merged_properties(m,['base','hinge'])
        for name in before:
            np.testing.assert_allclose(np.asarray(after[name]['inertia'])-before[name]['inertia'],.01*np.eye(3),atol=1e-12)
        np.testing.assert_allclose(m.dof_armature,0.)
        self.assertEqual(m.dof_damping[-1],1.)
        self.assertLess(report['after_full_inertia_max_error'],1e-12)

    def test_reject_unexplained_inertia_before_mutation(self):
        m,ck,runtime,probe,_=self.fixture()
        runtime['final_bodies'][0]['inertia'][0][1]+=.02
        original=m.body_inertia.copy()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'r.json').write_text(json.dumps(runtime));(p/'p.json').write_text(json.dumps(probe))
            with self.assertRaisesRegex(ValueError,'not verified'):
                align_runtime(m,ck,p/'r.json',p/'p.json')
        np.testing.assert_array_equal(m.body_inertia,original)


if __name__=='__main__':
    unittest.main()
