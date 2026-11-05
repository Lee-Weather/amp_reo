"""Validate the frozen B1 BC plan; real updates require an explicit --train flag."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from humanoid.amp.student_distillation import (verify_inputs, build_student, freeze_features,
    frozen_digest, state_digest, actor_features, checked_backward, fit_metrics,
    save_checkpoint, restore_checkpoint, trusted_load, sha256, checkpoint_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--train', action='store_true', help='Explicit authorization boundary for real optimizer steps')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    parser.add_argument('--checkpoint-dir', type=Path, help='Platform SDK directory containing only the verified input PT')
    args = parser.parse_args()
    if bool(args.resume) != bool(args.resume_sha256):
        parser.error('Resume requires both checkpoint path and independently recorded SHA256')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, data, meta, source, integrity = verify_inputs(plan, ROOT)
    checkpoint_dir = checkpoint_directory(args.output, args.checkpoint_dir, paths['checkpoint'])
    torch.set_num_threads(1)
    torch.manual_seed(plan['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_student(source).to(args.device)
    trainable = freeze_features(model)
    frozen = frozen_digest(model)
    original_digest = state_digest(model.state_dict())
    budget = plan['budget']
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=budget['learning_rate'])
    generator = torch.Generator().manual_seed(plan['seed'])
    observations = torch.as_tensor(data['student_observations'], device=args.device)
    labels = torch.as_tensor(data['label_actions'], device=args.device)
    default = torch.tensor(meta['default_dof_pos'], device=args.device)
    limits = torch.tensor(meta['hard_limits'], device=args.device)
    # Frozen branches can be cached exactly; validate cached and full inference before any update.
    features = actor_features(model, observations)
    with torch.no_grad():
        initial = model.actor(features).detach().clone()
        full = model.act_inference(observations)
        feature_error = float((initial-full).abs().max())
    if feature_error > 1e-6 or not torch.isfinite(initial).all().item():
        raise ValueError('Invalid frozen feature cache')
    scale, clip = meta['action_scale'], meta['action_clip']
    def metrics():
        with torch.no_grad():
            return fit_metrics(model.actor(features), labels, default, limits, scale, clip, initial)
    baseline = metrics()
    best = dict(step=0, raw_target_rmse_rad=baseline['raw_target_rmse_rad'], checkpoint=None)
    start_step = 0
    if args.resume:
        resumed = trusted_load(args.resume, args.resume_sha256)
        start_step, best = restore_checkpoint(resumed, model, optimizer, generator, plan, frozen)
    before = state_digest(model.state_dict())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    started = time.monotonic()
    updates = start_step
    report = dict(status='running' if args.train else 'dry_run', training_updates=0,
        restored_updates=start_step, source_sha256=sha256(paths['checkpoint']), dataset_integrity=integrity,
        initial_fit=baseline, trainable_names=trainable,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_sha256=frozen, cached_full_inference_max_error=feature_error,
        independent_validation_episodes=0, validation='one demonstration; fit metrics only',
        device=args.device, torch_version=torch.__version__, checkpoint_receipts={})
    try:
        # Dry-run performs one real-data forward/backward, but never optimizer.step or checkpoint save.
        stop = budget['updates'] if args.train else start_step+1
        for step in range(start_step+1, stop+1):
            if time.monotonic()-started > budget['max_wall_seconds']:
                raise ValueError('Pilot wall-time budget exceeded')
            indices = torch.randperm(len(labels), generator=generator)[:budget['batch_size']].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.actor(features[indices])
            loss = (scale*(prediction-labels[indices])).square().mean()
            gradient = checked_backward(model, loss, budget['max_grad_norm'])
            if not args.train:
                report.update(backward_loss_rad2=float(loss), gradient_norm_before_clip=gradient)
                break
            optimizer.step()
            updates = step
            report['training_updates'] = step-start_step
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError('Nonfinite parameter after update')
            current = metrics()
            if (current['raw_target_shift_rms_rad'] > budget['max_raw_target_shift_rms_rad']
                    or current['raw_target_rmse_rad'] > max(baseline['raw_target_rmse_rad']*2, 1e-4)):
                raise ValueError('Pilot divergence/target-shift guard triggered')
            if frozen_digest(model) != frozen:
                raise ValueError('Frozen feature/critic/std tensor changed')
            checkpoint = ('model_%d.pt' % step if args.checkpoint_dir is not None else 'student_step_%04d.pt' % step)
            if step % budget['save_interval'] == 0 or step == budget['updates']:
                if current['raw_target_rmse_rad'] < best['raw_target_rmse_rad']:
                    best = dict(step=step, raw_target_rmse_rad=current['raw_target_rmse_rad'],
                                checkpoint=str((checkpoint_dir/checkpoint).resolve()))
                save_checkpoint(checkpoint_dir/checkpoint, model, optimizer, generator, step, plan, source, frozen, best)
                report['checkpoint_receipts'][checkpoint] = sha256(checkpoint_dir/checkpoint)
                print('CHECKPOINT_SAVED '+json.dumps(dict(iteration=step, path=str((checkpoint_dir/checkpoint).resolve()),
                    bytes=(checkpoint_dir/checkpoint).stat().st_size, sha256=report['checkpoint_receipts'][checkpoint])), flush=True)
                (args.output/'checkpoint_receipts.json').write_text(
                    json.dumps(report['checkpoint_receipts'], indent=2)+'\n', encoding='utf-8')
            with (args.output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, batch_loss_rad2=float(loss), gradient_norm=gradient, **current), allow_nan=False)+'\n')
            if step % 10 == 0:
                print('BC_STEP '+json.dumps(dict(step=step, **current)), flush=True)
        if not args.train and (state_digest(model.state_dict()) != before or optimizer.state):
            raise ValueError('Dry-run unexpectedly changed model or optimizer state')
        final_fit = metrics()
        report.update(status='completed' if args.train else 'prepared_not_started', total_updates=updates,
            best=best, final_fit=final_fit, model_unchanged=state_digest(model.state_dict()) == before,
            original_source_state_sha256=original_digest, optimizer_state_entries=len(optimizer.state),
            frozen_exact=frozen_digest(model) == frozen,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
        report['offline_fit_passed'] = bool(args.train and best['checkpoint'] is not None
            and best['raw_target_rmse_rad'] <= .06
            and best['raw_target_rmse_rad'] <= baseline['raw_target_rmse_rad']*.7)
    except Exception as error:
        report.update(status='failed', error=str(error), total_updates=updates)
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('BC_RESULT '+json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
