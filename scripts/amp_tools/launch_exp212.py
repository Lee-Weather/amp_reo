"""One private nominal-startup run on the currently authorized account."""
import launch_exp210 as p

p.EXPERIMENT='exp2.12'
p.TASK='x1_amp_exp212'
p.ENTRY='train_nominal_amp.py'
p.TRANSFER_MODULE='exp212_transfer.py'
p.REFINEMENT='exp212_nominal_startup_v1'
p.NAME='exp2.12-nominal-startup-4096-seed5-plus1000-entryfix'
p.BRANCH='codex/exp2-12-startup-fix'
p.PROJECT_NAME='X1 29DOF AMP exp2.12 nominal startup coverage'
p.DIRECTORY=p.ROOT/'czy/exp1/records/exp2.12_evaluation'
p.STATE=p.ROOT/'czy/exp1/records/exp2.12_training.json'
p.DESCRIPTION='Full13500 to14500 +1000 updates;25percent environments exact standing reset,no observation noise,forward-only command,friction0.8,contact_offset1mm;remaining75percent exp2.11 domain/reset/noise/commands. Preserve source rewards,yaw-0.10,AMP0.01,Actor/Critic,D13300,bothAdam,original anchor,PD,network,20s episode and phase.4096seed5,24rollout2x4;first50,save100,best,original guards. Target calibrated MuJoCo60s forward without falls or resets; no GPU short training.'
p.CHANGE_FIELDS=dict(nominal_startup_fraction=.25,nominal_startup_envs=1024,
    replaces_zero_update_task='TASK_20261104_180',
    contact_offset_range_m=[.001,.020],contact_populations=dict(near=.5,original=.25,uniform=.25),
    source_selection='source13500 better calibrated nominal survival than exp210/211; not statistically optimal',
    acceptance='calibrated_MuJoCo_60s_no_fall_no_reset_native_speed')

if __name__=='__main__':p.main()
