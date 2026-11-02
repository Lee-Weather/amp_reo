"""Reuse a successful7000 native-speed replay; never synthesize its history."""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]

def main():
    path=ROOT/'czy/exp1/records/exp2.1_evaluation/source_native_bank_input.csv'
    source=ROOT/'czy/data/exp1.17/model_7000.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    names=checkpoint['metadata']['dof_names']
    scales=checkpoint['config']['normalization']['obs_scales']
    default=checkpoint['config']['init_state']['default_joint_angles']
    rows=list(csv.DictReader(path.open()))
    def vector(row,prefix):return [float(row[prefix+'_'+n]) for n in names]
    def frame(row):
        return ([float(row[k])*s for k,s in zip(('cmd_vx','cmd_vy','cmd_wz'),(scales['lin_vel'],scales['lin_vel'],scales['ang_vel']))]
            +[float(row[k])*scales['ang_vel'] for k in ('wx','wy','wz')]
            +[float(row[k]) for k in ('gx','gy','gz')]
            +[(float(row['q_'+n])-default[n])*scales['dof_pos'] for n in names]
            +[x*scales['dof_vel'] for x in vector(row,'dq')]+vector(row,'action')
            +[math.sin(2*math.pi*float(row['reference_phase'])),math.cos(2*math.pi*float(row['reference_phase']))])
    candidates=[]
    for i in range(66,len(rows)-100):
        window=rows[i-65:i+101]
        if (all(r['episode_id']==rows[i]['episode_id'] and r['done']=='0' for r in window)
            and all(float(r['cmd_vx'])>.59 and r['command_transition']=='0' for r in window)
            and float(rows[i]['gz'])<-.90): candidates.append(i)
    assert len(candidates)>=64
    selected=[]
    for b in range(64):
        pool=[i for i in candidates if i not in selected]
        i=min(pool,key=lambda i:abs(((float(rows[i]['reference_phase'])-b/64+.5)%1)-.5))
        selected.append(i)
    def state(row):return [float(row[k]) for k in ('x','y','z','qx','qy','qz','qw','world_vx','world_vy','world_vz','world_wx','world_wy','world_wz')]+vector(row,'q')+vector(row,'dq')
    quiet=[i for i in range(66,len(rows)) if all(r['episode_id']==rows[i]['episode_id'] and float(r['cmd_vx'])==0 and r['done']=='0' for r in rows[i-65:i+1])]
    quiet=quiet[::max(1,len(quiet)//12)][:12]
    payload=dict(version=1,source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_code_commit='7e96e4e51d1a2e12e309b24bb8288a718b659511',dof_names=names,
        control_dt=.02,frames=66,frame_dim=98,selection='64 phase-balanced full-speed states with66 real history frames and2s verified nonterminal continuation',
        states=[state(rows[i]) for i in selected],phases=[float(rows[i]['reference_phase']) for i in selected],
        histories=[[frame(r) for r in rows[i-65:i+1]] for i in selected],
        amp_states=[[state(r) for r in rows[i-2:i+1]] for i in selected],
        quiet_histories=[[frame(r)[:-2]+[0.,0.] for r in rows[i-65:i+1]] for i in quiet],
        source_rows=selected)
    output=ROOT/'humanoid/amp/reference_data/exp22_policy_state_bank.json'
    output.write_bytes((json.dumps(payload,separators=(',',':'))+'\n').encode('utf-8'))
    print(json.dumps(dict(states=len(selected),quiet_anchors=len(quiet),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
