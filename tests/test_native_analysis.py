"""Analyze synthetic native-reference CSVs without pretending they are rollouts."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_exp20 import configuration20
from test_exp117 import grounded_library
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.motion import quat_rotate

ROOT = Path(__file__).resolve().parents[1]


class NativeAnalysisTests(unittest.TestCase):
    def test_forward_analysis_uses_corrected_native_reference_and_safe_windows(self):
        torch.set_num_threads(1)
        settings = configuration20()['X1AMPExp20Cfg']()
        path = ROOT/'humanoid/amp/reference_data/exp20_native_correction.json'
        correction = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(correction,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        lib = grounded_library()
        reference = NativeCycle(lib,contact_correction=correction)
        time = (torch.arange(3000)+1)*.02
        state = reference.state(time/1.26)
        positions = lib.fk.key_positions(state[:,13:42])[:,:2]
        world = state[:,None,:3]+quat_rotate(state[:,None,3:7].expand(-1,2,-1),positions)
        hashes = {name:hashlib.sha256((ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'/(name+'.pkl')).read_bytes()).hexdigest() for name in lib.names}
        cfg = dict(key_body_names=lib.key_body_names,history_steps=3,motion_weights=settings.amp.motion_weights,
            limit_tolerance=.05,ground_clearance=.002,standing_clip=settings.amp.standing_clip,standing_frame=11,
            native_clip=settings.amp.native_clip,native_first_frame=27,native_last_frame=90,
            native_correction_file=settings.amp.native_correction_file)
        pseudo_checkpoint = dict(metadata=dict(dof_names=lib.dof_names,policy_dt=.02,training=dict(
            curriculum='native_full_body_v1',reference_source_sha256=hashes,native_reference=dict(correction_sha256=digest))),
            config=dict(amp=cfg,commands=dict(native_speed=settings.commands.native_speed)))
        sys.path.insert(0,str(ROOT/'scripts/amp_tools'))
        try:
            spec = importlib.util.spec_from_file_location('native_forward_analysis',ROOT/'scripts/amp_tools/analyze_forward_gait.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory(prefix='native_analysis_test_') as folder:
                folder = Path(folder)
                model,data,output = folder/'synthetic_metadata.pt',folder/'synthetic.csv',folder/'report'
                torch.save(pseudo_checkpoint,model)
                with data.open('w',newline='') as stream:
                    writer = None
                    for i in range(3000):
                        row = dict(time_s=float(time[i]),episode_id=i//1001,episode_step=i%1001+1,done=int(i%1001==1000),
                            command_requested=settings.commands.native_speed,command_applied=settings.commands.native_speed,
                            base_vel_x=settings.commands.native_speed,base_euler_y=0.,foot_force_l=180.,foot_force_r=180.)
                        row.update(zip(('x','y','z','qx','qy','qz','qw'),state[i,:7].tolist()))
                        row.update(('q_'+name,float(state[i,13+j])) for j,name in enumerate(lib.dof_names))
                        for j,side in enumerate(('l','r')):
                            row.update(('foot_'+axis+'_'+side,float(world[i,j,k])) for k,axis in enumerate('xyz'))
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                with patch.object(sys,'argv',['analyze_forward_gait.py','--model',str(model),'--csv',str(data),'--output-dir',str(output)]),contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                report = json.loads((output/'forward_gait_analysis.json').read_text())
                self.assertAlmostEqual(report['reference_period_s'],1.26,places=6)
                self.assertEqual(set(report['policy_windows']),{'6-18','26-38','46-58'})
                self.assertIn('native',report['definitions']['reference'])
                self.assertGreater(report['reference']['feet']['left']['sole_p95_m'],.02)
                self.assertTrue((output/'forward_gait_comparison.png').exists())
        finally:
            sys.path.pop(0)


if __name__=='__main__': unittest.main()
