"""Fix only the zero-update startup import order; no algorithm or budget change."""
import publish_exp212
p=publish_exp212.p
p.PARENT='f20284ba2cddc1fd14d2c310b69dcfddec7c8c42'
p.BRANCH='codex/exp2-12-startup-fix'
p.OUTPUT_DIRECTORY=p.ROOT/'czy/exp1/records/exp2.12_evaluation/startup_fix'
p.FILES=['humanoid/amp/nominal_startup.py','tests/test_exp212.py']
p.REGISTRATION=''
p.MESSAGE=b'Fix exp2.12 entry import order before Isaac Gym; assert import contract in fresh process\n'

if __name__=='__main__':p.main()
