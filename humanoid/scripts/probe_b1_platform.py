"""Same-platform B1 teacher/source/student controls; inference only, seed 5."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.scripts.train_b1_platform import unpack_inputs


def download(encoded, expected, maximum=64*1024*1024):
    try:
        url = base64.urlsafe_b64decode(encoded).decode()
        if not url.startswith('https://'):
            raise ValueError()
        with urllib.request.urlopen(url, timeout=90) as response:
            blob = response.read(maximum+1)
        if len(blob) > maximum or hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError()
        return blob
    except Exception:
        raise RuntimeError('Private diagnostic input download/SHA failed; URL withheld') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_url_b64', required=True)
    parser.add_argument('--input_sha256', required=True)
    parser.add_argument('--checkpoint_url_b64', required=True)
    parser.add_argument('--checkpoint_sha256', required=True)
    parser.add_argument('--takeover', action='store_true', help='Four scheduled teacher recovery cases, still inference only')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['PYTHONPATH'] = str(ROOT)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    plan_path = ROOT/'configs/b1_bc_pilot_platform.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    unpack_inputs(download(args.input_url_b64, args.input_sha256), plan, ROOT)
    checkpoint_dir = (ROOT/plan['inputs']['checkpoint']['path']).parent
    trained = checkpoint_dir/'input_trained_model_200.pt'
    with trained.open('xb') as stream:
        stream.write(download(args.checkpoint_url_b64, args.checkpoint_sha256))
    (ROOT/plan['inputs']['checkpoint']['path']).touch()
    work = ROOT/('logs/exp213_takeover' if args.takeover else 'logs/exp213_controls')
    work.mkdir(exist_ok=False)
    state = dict(schema='x1_b1_platform_controls_v1', training_updates=0, seed=5, status='running',
        cases=(['step%d_%s' % (step, history) for step in (20, 40) for history in ('target_equivalent', 'shadow')]
               if args.takeover else ['teacher', 'source_student', 'trained_student']),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        checkpoint_sha256=args.checkpoint_sha256, input_sha256=args.input_sha256)
    def record():
        (work/'state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    def run(case, arguments):
        print('CONTROL_START '+json.dumps(dict(case=case, training_updates=0)), flush=True)
        with (work/(case+'.log')).open('x', encoding='utf-8') as output:
            process = subprocess.Popen([sys.executable]+arguments, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError('Control case exceeded 900 seconds: '+case) from None
        print((work/(case+'.log')).read_text(encoding='utf-8', errors='replace'), flush=True)
        if code:
            raise RuntimeError('Control subprocess failed: '+case)
        summary = json.loads((work/case/'summary.json').read_text())
        print('CONTROL_RESULT '+json.dumps(dict(case=case, summary=summary)), flush=True)
        return summary
    record()
    error = None
    try:
        inputs = {key: str(ROOT/item['path']) for key, item in plan['inputs'].items()}
        if args.takeover:
            summaries = {}
            for step in (20, 40):
                for history in ('target_equivalent', 'shadow'):
                    case = 'step%d_%s' % (step, history)
                    summaries[case] = run(case, ['scripts/amp_tools/probe_b1_takeover.py', '--plan', str(plan_path),
                        '--checkpoint', str(trained), '--checkpoint-sha256', args.checkpoint_sha256,
                        '--takeover-step', str(step), '--history', history, '--output', str(work/case)])
            state.update(status='completed', takeover_results=summaries)
        else:
            teacher = run('teacher', ['scripts/amp_tools/collect_wbt_student.py', '--bundle', inputs['teacher_bundle'],
                '--teacher-lock', inputs['teacher_lock'], '--condition-profile', inputs['profile'], '--output', str(work/'teacher')])
            source = run('source_student', ['scripts/amp_tools/eval_b1_student.py', '--plan', str(plan_path),
                '--output', str(work/'source_student')])
            student = run('trained_student', ['scripts/amp_tools/eval_b1_student.py', '--plan', str(plan_path),
                '--checkpoint', str(trained), '--checkpoint-sha256', args.checkpoint_sha256,
                '--output', str(work/'trained_student')])
            state.update(status='completed', teacher_completed=teacher['metadata']['completed_reference'],
                source_steps=source['steps'], trained_steps=student['steps'],
                source_reason=source['termination_reason'], trained_reason=student['termination_reason'])
    except Exception as exception:
        error = str(exception)
        state.update(status='failed', error=error)
    finally:
        record()
        import torch
        files = {path.relative_to(work).as_posix(): path.read_bytes() for path in work.rglob('*')
                 if path.is_file() and path.suffix in ('.json', '.log', '.npz')}
        artifact = checkpoint_dir/'model_isaac_csv.pt'
        with artifact.open('xb') as stream:
            torch.save(dict(schema='x1_b1_controls_evidence_v1', files=files, state=state), stream)
        print('CONTROL_FINAL '+json.dumps(state), flush=True)
        print('EVIDENCE_SAVED '+json.dumps(dict(bytes=artifact.stat().st_size,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())), flush=True)
        time.sleep(90)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
