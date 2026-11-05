"""Verify the first output and actual nominal-startup runtime evidence."""
import inspect_exp210 as p
import json
p.EXPERIMENT='exp2.12'

if __name__=='__main__':
    directory=p.ROOT/'czy/exp1/records/exp2.12_evaluation'
    proof=json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert proof['verified'] and proof['nominal_startup_verified'] and proof['contact_properties_verified']
    p.main()
    path=directory/'first_checkpoint_inspection.json';report=json.loads(path.read_text())
    report.update(nominal_startup_verified=True,source_rewards_unchanged=True,
        nominal_startup_markers={k:v for k,v in proof['startup_markers'].items() if k.startswith('NOMINAL_STARTUP_')})
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    state_path=p.ROOT/'czy/exp1/records/exp2.12_training.json';state=json.loads(state_path.read_text())
    state['first_checkpoint_inspection']=report
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
