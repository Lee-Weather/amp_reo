"""Publish the actual inherited reset hook and exact remaining-budget recovery."""
import publish_exp212
p=publish_exp212.p
p.PARENT='937f329d57ae9c4a4a06665217e9ccf893821806'
p.BRANCH='codex/exp2-12-reset-fix'
p.OUTPUT_DIRECTORY=p.ROOT/'czy/exp1/records/exp2.12_evaluation/reset_fix'
p.FILES=['humanoid/envs/x1/x1_amp_exp22_env.py','humanoid/amp/nominal_startup.py',
    'humanoid/amp/exp212_transfer.py','tests/test_exp212.py']
p.REGISTRATION=''
p.MESSAGE=b'Fix exp2.12 actual policy-bank reset path; recover verified13550 with remaining950 updates\n'

if __name__=='__main__':p.main()
