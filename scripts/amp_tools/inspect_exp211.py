"""First contact-domain checkpoint must preserve full source state continuity."""
import inspect_exp210 as p
import json
p.EXPERIMENT='exp2.11'

if __name__=='__main__':
    directory=p.ROOT/'czy/exp1/records/exp2.11_evaluation'
    proof=json.loads((directory/'first_checkpoint_verification.json').read_text())
    contact=proof['startup_markers']['CONTACT_RANDOMIZATION_READY']
    assert proof['verified'] and proof['contact_properties_verified'] and contact['verified']
    assert abs(contact['assigned_min_m']-.001)<1e-7 and contact['assigned_max_m']<=.0200001
    assert contact['near_fraction']==.5 and contact['original_fraction']==.25
    assert len(contact['sampled_actor_checks'])==3
    p.main()
    path=directory/'first_checkpoint_inspection.json';report=json.loads(path.read_text())
    report.update(contact_properties=contact,source_rewards_unchanged=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    state_path=p.ROOT/'czy/exp1/records/exp2.11_training.json';state=json.loads(state_path.read_text())
    state['first_checkpoint_inspection']=report
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
