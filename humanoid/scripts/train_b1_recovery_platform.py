"""Private-input platform adapter for the masked recovery BC round."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/b1_recovery_platform.json'


def unpack_inputs(blob, plan, root):
    expected = {item['path']: item['sha256'] for item in plan['inputs'].values()}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError('Private input bundle differs from the frozen plan')
        for member in members:
            path = (root/member.name).resolve()
            if not member.isfile() or root.resolve() not in path.parents or member.size > 64*1024*1024:
                raise ValueError('Unsafe private input member')
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError('Private input member SHA mismatch')
            if path.exists():
                raise FileExistsError('Private input destination already exists')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as output:
                output.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_url_b64', required=True)
    parser.add_argument('--input_sha256', required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONPATH'] = str(ROOT)
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    try:
        url = base64.urlsafe_b64decode(args.input_url_b64).decode()
        if not url.startswith('https://'):
            raise ValueError()
        with urllib.request.urlopen(url, timeout=90) as response:
            blob = response.read(64*1024*1024+1)
        if len(blob) > 64*1024*1024 or hashlib.sha256(blob).hexdigest() != args.input_sha256:
            raise ValueError()
    except Exception:
        raise RuntimeError('Private input download or SHA verification failed; URL withheld') from None
    unpack_inputs(blob, plan, ROOT)
    initial = ROOT/plan['inputs']['checkpoint']['path']
    checkpoint_dir = initial.parent
    initial.touch()
    print('CHECKPOINT_INPUT_READY '+json.dumps(dict(path=str(initial), bytes=initial.stat().st_size,
        sha256=plan['inputs']['checkpoint']['sha256'], bundle_sha256=args.input_sha256)), flush=True)
    work = ROOT/'logs/exp214_recovery'
    work.mkdir(parents=True, exist_ok=False)
    state = dict(schema='x1_b1_recovery_platform_run_v1', status='preflight', training_updates=0,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(), seed=5,
        input_bundle_sha256=args.input_sha256)
    def record():
        (work/'state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    record()
    def run(arguments, log, timeout):
        # Stream useful child markers while keeping private download arguments out of logs.
        with (work/log).open('x', encoding='utf-8') as output:
            process = subprocess.Popen([sys.executable]+arguments, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError('Pilot child exceeded its fixed wall-time limit') from None
        text = (work/log).read_text(encoding='utf-8', errors='replace')
        print(text, flush=True)
        if code:
            raise RuntimeError('Pilot child failed: '+log)
    failed = None
    try:
        run(['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_playback.py', '-v'], 'unit_tests.log', 120)
        run(['scripts/amp_tools/train_b1_recovery.py', '--plan', str(PLAN), '--output', str(work/'dry_run'),
             '--device', 'cuda:0', '--checkpoint-dir', str(checkpoint_dir)], 'dry_run.log', 120)
        dry = json.loads((work/'dry_run/summary.json').read_text())
        if not (dry['model_unchanged'] and dry['optimizer_state_entries'] == 0 and dry['training_updates'] == 0):
            raise ValueError('Platform real-data dry-run changed student state')
        print('PLATFORM_PREFLIGHT_OK '+json.dumps(dict(training_updates=0, model_unchanged=True)), flush=True)
        if not args.train:
            state['status'] = 'prepared_not_started'
            return
        state['status'] = 'training'
        record()
        run(['scripts/amp_tools/train_b1_recovery.py', '--plan', str(PLAN), '--output', str(work/'training'),
             '--device', 'cuda:0', '--checkpoint-dir', str(checkpoint_dir), '--train'], 'training.log', 660)
        training = json.loads((work/'training/summary.json').read_text())
        state['training_updates'] = training['total_updates']
        state['offline_fit_passed'] = training['offline_fit_passed']
        state['warm_start_updates'] = 200
        checkpoint = training['best']['checkpoint']
        if checkpoint is None:
            raise ValueError('No saved candidate improved the balanced recovery fit')
        checkpoint = Path(checkpoint)
        expected = training['checkpoint_receipts'][checkpoint.name]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError('Saved student receipt mismatch')
        state.update(status='evaluating', selected_checkpoint=checkpoint.name, selected_sha256=expected)
        record()
        run(['scripts/amp_tools/eval_b1_recovery.py', '--plan', str(PLAN), '--checkpoint', str(checkpoint),
             '--checkpoint-sha256', expected, '--output', str(work/'evaluation')], 'evaluation.log', 900)
        replay = json.loads((work/'evaluation/summary.json').read_text())
        state.update(status='completed', closed_loop_passed=replay['passed'],
                     passed=bool(training['offline_fit_passed'] and replay['passed']))
    except Exception as error:
        failed = str(error)
        state.update(status='failed', error=failed)
        summary = work/'training/summary.json'
        if summary.exists():
            state['training_updates'] = json.loads(summary.read_text()).get('total_updates', 0)
    finally:
        record()
        # All PTs share the first SDK-discovered directory. This is an evidence archive, not a policy.
        import torch
        files = {path.relative_to(work).as_posix(): path.read_bytes() for path in work.rglob('*')
                 if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.npz')}
        artifact = checkpoint_dir/'model_isaac_csv.pt'
        with artifact.open('xb') as stream:
            torch.save(dict(schema='x1_b1_recovery_platform_evidence_v1', files=files, state=state), stream)
        print('B1_RECOVERY_PLATFORM_RESULT '+json.dumps(state), flush=True)
        print('EVIDENCE_SAVED '+json.dumps(dict(path=str(artifact), bytes=artifact.stat().st_size,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())), flush=True)
        print('SDK_UPLOAD_GRACE_SECONDS 90', flush=True)
        time.sleep(90)
    if failed:
        raise RuntimeError(failed)
    if state.get('passed') is False:
        sys.exit(3)


if __name__ == '__main__':
    main()
