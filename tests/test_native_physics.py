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


class NativePhysicsTest(unittest.TestCase):
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
