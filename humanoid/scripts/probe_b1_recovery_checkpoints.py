"""Evaluate fixed recovery-BC checkpoints in separate Gym processes; no training."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.scripts.probe_b1_platform import download
from humanoid.scripts.train_b1_platform import unpack_inputs


def decode_models(encoded, expected):
    """Bind runtime download locations to the committed checkpoint identities."""
    try:
        rows = json.loads(base64.urlsafe_b64decode(encoded).decode())
        if (not isinstance(rows, list) or len(rows) != len(expected)
                or [str(row['step']) for row in rows] != list(expected)):
            raise ValueError()
        for row in rows:
            if row['sha256'] != expected[str(row['step'])] or not row['url'].startswith('https://'):
                raise ValueError()
        return rows
    except Exception:
        raise ValueError('Checkpoint download manifest differs from the fixed identities; URLs withheld') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_url_b64', required=True)
    parser.add_argument('--input_sha256', required=True)
    parser.add_argument('--checkpoint_url_b64', required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['PYTHONPATH'] = str(ROOT)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    config = json.loads((ROOT/'configs/b1_recovery_checkpoint_controls.json').read_text(encoding='utf-8'))
    plan_path = ROOT/'configs/b1_recovery_platform.json'
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != config['training_plan_sha256']:
        raise ValueError('Frozen recovery training plan changed')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    for relative, expected in dict(plan['source_sha256'], **config['diagnostic_source_sha256']).items():
        actual = hashlib.sha256((ROOT/relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('Frozen inference source changed: '+relative)
    if args.input_sha256 != config['input_bundle_sha256'] or config['training_updates'] != 0:
        raise ValueError('Input bundle or zero-update diagnostic contract changed')
    models = decode_models(args.checkpoint_url_b64, config['checkpoints'])
    unpack_inputs(download(args.input_url_b64, args.input_sha256), plan, ROOT)
    checkpoint_dir = (ROOT/plan['inputs']['checkpoint']['path']).parent
    (ROOT/plan['inputs']['checkpoint']['path']).touch()
    for row in models:
        row['path'] = checkpoint_dir/('input_recovery_step%d.pt' % row['step'])
        encoded = base64.urlsafe_b64encode(row['url'].encode()).decode()
        with row['path'].open('xb') as stream:
            stream.write(download(encoded, row['sha256']))
    work = ROOT/'logs/exp214_checkpoint_controls'
    work.mkdir(parents=True, exist_ok=False)
    state = dict(schema='x1_b1_checkpoint_controls_v1', training_updates=0, seed=5,
        source_task=config['source_task'], status='running', results={})
    def record():
        (work/'state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    record()
    error = None
    try:
        for row in models:
            case = 'step%d' % row['step']
            print('CONTROL_START '+json.dumps(dict(case=case, training_updates=0)), flush=True)
            with (work/(case+'.log')).open('x', encoding='utf-8') as output:
                process = subprocess.Popen([sys.executable, 'scripts/amp_tools/eval_b1_recovery.py',
                    '--plan', str(plan_path), '--checkpoint', str(row['path']), '--checkpoint-sha256', row['sha256'],
                    '--output', str(work/case)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    process.terminate(); process.wait(timeout=10)
                    raise RuntimeError('Checkpoint replay exceeded 900 seconds: '+case) from None
            print((work/(case+'.log')).read_text(encoding='utf-8', errors='replace'), flush=True)
            if code:
                raise RuntimeError('Checkpoint replay process failed: '+case)
            summary = json.loads((work/case/'summary.json').read_text())
            if (summary['checkpoint_sha256'] != row['sha256'] or summary['training_updates'] != 0
                    or summary['teacher_action_queries'] != 0 or not summary['model_unchanged']
                    or hashlib.sha256(row['path'].read_bytes()).hexdigest() != row['sha256']):
                raise ValueError('Replay modified or substituted the checkpoint: '+case)
            state['results'][case] = summary
            record()
            print('CONTROL_RESULT '+json.dumps(dict(case=case, summary=summary)), flush=True)
        state['status'] = 'completed'
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
            torch.save(dict(schema='x1_b1_checkpoint_controls_evidence_v1', state=state, files=files), stream)
        print('CONTROL_FINAL '+json.dumps(state), flush=True)
        print('EVIDENCE_SAVED '+json.dumps(dict(bytes=artifact.stat().st_size,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())), flush=True)
        print('SDK_UPLOAD_GRACE_SECONDS 90', flush=True)
        time.sleep(90)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
