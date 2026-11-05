"""One private contact-domain run on the user's current account."""
import launch_exp210 as p

p.EXPERIMENT='exp2.11'
p.TASK='x1_amp_exp211'
p.ENTRY='train_contact_amp.py'
p.TRANSFER_MODULE='exp211_transfer.py'
p.REFINEMENT='exp211_contact_only_v1'
p.NAME='exp2.11-contact-offset-4096-seed5-plus1000'
p.BRANCH='codex/exp2-11'
p.PROJECT_NAME='X1 29DOF AMP exp2.11 sim2sim contact robustness'
p.DIRECTORY=p.ROOT/'czy/exp1/records/exp2.11_evaluation'
p.STATE=p.ROOT/'czy/exp1/records/exp2.11_training.json'
p.DESCRIPTION='Full13500 to14500 +1000updates; only broaden PhysX shape contact_offset:50percent1mm,25percentoriginal10mm,25percentuniform1-20mm,rest_offset0. Source13500 chosen by calibrated MuJoCo replay, not latest iteration. Preserve source rewards,yaw-0.10,AMP0.01,Actor/Critic,D13300,bothAdam,anchor,PD,phase and friction0.65-0.95. 4096seed5 24rollout2x4;save100,first50,best;original regression and5.5h guards. Success requires calibrated MuJoCo60seconds without falls at native speed.'
p.CHANGE_FIELDS=dict(contact_offset_range_m=[.001,.020],contact_populations=dict(near=.5,original=.25,uniform=.25),
    source_selection='longest calibrated nominal replay among13500and14500; not a statistical optimum',
    acceptance='calibrated_MuJoCo_60s_no_fall_no_reset_native_speed')

if __name__=='__main__':p.main()
