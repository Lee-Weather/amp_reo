"""CPU source transfer, conservative update tests and B1/B2 join screening."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import os
import sys
import unittest

import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['EXP21_SOURCE_MODEL']=str(a.source.resolve())
    from test_exp21 import fixture21
    from humanoid.amp.native_reference import NativeCycle
    torch.set_num_threads(1);torch.manual_seed(5)
    source=torch.load(a.source,map_location='cpu',weights_only=False)
    env,cfg=fixture21(source);lib=env.motion_library
    correction=json.loads((ROOT/'humanoid/amp/reference_data/exp20_native_correction.json').read_text())
    ref=NativeCycle(lib,contact_correction=correction)
    phases=torch.arange(128)/128
    native=ref.state(phases)
    joins={}
    for name,first,last in [('F1_B1_stand_to_walk',60,180),('F1_B2_walk_to_stand',0,190)]:
        i=lib.names.index(name);states=lib.frames[i,first:last+1]
        dq=(states[:,None,13:42]-native[None,:,13:42]).abs()
        dv=(states[:,None,42:]-native[None,:,42:]).abs()
        cost=(dq/.15).square().mean(-1)+(dv/1.5).square().mean(-1)
        frame,phase=divmod(int(cost.argmin()),len(phases))
        qmax=float(dq[frame,phase].max());vmax=float(dv[frame,phase].max())
        joins[name]=dict(candidate_frame=first+frame,native_phase=float(phases[phase]),
            max_joint_gap_rad=qmax,max_dq_gap_rad_s=vmax,
            q_dq_screen_passed=qmax<=.15 and vmax<=1.5,
            complete_contact_root_interface_passed=False)
    # A q/dq nearest neighbor alone never authorizes using the full transition.
    test_stream=io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_exp21.py')
        result=unittest.TextTestRunner(stream=test_stream,verbosity=2).run(suite)
    report=dict(source_iteration=source['iter'],new_initial_iteration=0,
        source_sha256=cfg['runner']['policy_transfer_source_sha256'],
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        software_passed=result.wasSuccessful(),reference_acceptance=ref.audit(),
        transition_join_screen=joins,b1_b2_replacement_enabled=False,
        transition_decision='No complete support/root interface has been validated: keep the existing2s smooth ramp only for the200-update low-weight validation batch; do not claim B1/B2 or final standing-start acceptance.',
        initial_rsi_fraction=.7,critic_only_updates=50,std_frozen_through=200,
        initial_guide_raw_amp=[.1,.1,0.],num_envs=4096,seed=5,new_update_budget=200,
        gpu_simulation_run=False,ready_for_gpu_probe=result.wasSuccessful())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful(): print(test_stream.getvalue());raise RuntimeError('exp2.1 CPU tests failed')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
