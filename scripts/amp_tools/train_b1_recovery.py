"""Bounded masked recovery BC; a dry-run never performs optimizer updates."""
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

from humanoid.amp.student_distillation import (build_student, freeze_features, frozen_digest,
    state_digest, actor_features, checked_backward, fit_metrics, sha256)
from humanoid.amp.recovery_distillation import (verify_inputs, sample_balanced, masked_loss,
    SCHEMA, SEMANTICS, WARM_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, data, meta, source, integrity = verify_inputs(plan, ROOT)
    checkpoint_dir = args.checkpoint_dir or args.output
    if args.checkpoint_dir is not None:
        allowed = {paths['checkpoint'].resolve(), paths['warm_start'].resolve()}
        if (checkpoint_dir.resolve() != paths['warm_start'].parent
                or {path.resolve() for path in checkpoint_dir.glob('*.pt')} != allowed):
            raise ValueError('SDK directory must contain exactly the two verified input checkpoints')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    torch.set_num_threads(1)
    torch.manual_seed(5)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_student(source).to(args.device)
    trainable = freeze_features(model)
    before, frozen = state_digest(model.state_dict()), frozen_digest(model)
    budget = plan['budget']
    # Fresh Adam: the old PPO, discriminator and previous BC optimizer are not restored.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=budget['learning_rate'])
    generator = torch.Generator().manual_seed(5)
    observations = torch.as_tensor(data['student_observations'], device=args.device)
    labels = torch.as_tensor(data['label_actions'], device=args.device)
    mask = torch.as_tensor(data['label_mask'], device=args.device)
    episodes = torch.as_tensor(data['episode_id'], device=args.device)
    default = torch.tensor(meta['default_dof_pos'], device=args.device)
    limits = torch.tensor(meta['hard_limits'], device=args.device)
    scale, clip = meta['action_scale'], meta['action_clip']
    features = actor_features(model, observations)
    with torch.no_grad():
        initial = model.actor(features).detach().clone()
        feature_error = float((initial-model.act_inference(observations)).abs().max())
    if feature_error > 1e-6 or not torch.isfinite(initial).all().item():
        raise ValueError('Recovery cached/full inference mismatch or nonfinite warm start')
    def metrics():
        with torch.no_grad():
            prediction = model.actor(features)
            groups = {name: fit_metrics(prediction[valid], labels[valid], default, limits, scale, clip, initial[valid])
                      for name, valid in (('all', mask), ('nominal', mask & (episodes == 0)),
                          ('recovery', mask & (episodes == 1)),
                          ('early_recovery', mask & (episodes == 1) & (torch.as_tensor(data['reference_step'], device=args.device) < 70)))}
            groups['balanced_rmse_rad'] = ((groups['nominal']['raw_target_rmse_rad']**2+
                                           groups['recovery']['raw_target_rmse_rad']**2)/2)**.5
            return groups
    baseline = metrics()
    best = dict(step=0, balanced_rmse_rad=baseline['balanced_rmse_rad'], checkpoint=None, fit=baseline)
    report = dict(schema=SCHEMA, status='running' if args.train else 'dry_run', training_updates=0,
        total_updates=0, warm_start_updates=200, warm_start_sha256=WARM_SHA,
        recovery_sha256=sha256(paths['recovery']), dataset_integrity=integrity,
        initial_fit=baseline, cached_full_inference_max_error=feature_error,
        trainable_names=trainable, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_sha256=frozen, independent_validation_episodes=0,
        validation='Two correlated training episodes; fit metrics are not validation.',
        optimizer_fresh=True, initial_optimizer_state_entries=len(optimizer.state),
        device=args.device, torch_version=torch.__version__, checkpoint_receipts={})
    started = time.monotonic()
    try:
        for step in range(1, (budget['updates'] if args.train else 1)+1):
            if time.monotonic()-started > budget['max_wall_seconds']:
                raise ValueError('Recovery training wall-time budget exceeded')
            indices = sample_balanced(data['label_mask'], data['episode_id'], generator).to(args.device)
            if not mask[indices].all().item():
                raise ValueError('Unlabelled context entered the training batch')
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model.actor(features[indices]), labels[indices], mask[indices], scale)
            gradient = checked_backward(model, loss, budget['max_grad_norm'])
            if not args.train:
                report.update(backward_loss_rad2=float(loss), gradient_norm_before_clip=gradient)
                break
            optimizer.step()
            report['training_updates'] = report['total_updates'] = step
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError('Nonfinite recovery parameter')
            current = metrics()
            if (current['all']['raw_target_shift_rms_rad'] > budget['max_raw_target_shift_rms_rad']
                    or current['balanced_rmse_rad'] > baseline['balanced_rmse_rad']*2
                    or frozen_digest(model) != frozen):
                raise ValueError('Recovery divergence/frozen/target-shift guard triggered')
            if step % budget['save_interval'] == 0:
                path = checkpoint_dir/('model_%d.pt' % step)
                # Preserve nominal fit while choosing a balanced fit improvement. Closed-loop is evaluated separately.
                if (current['nominal']['raw_target_rmse_rad'] <= .06
                        and current['balanced_rmse_rad'] < best['balanced_rmse_rad']):
                    best = dict(step=step, balanced_rmse_rad=current['balanced_rmse_rad'],
                                checkpoint=str(path.resolve()), fit=current)
                metadata = json.loads(json.dumps(source['metadata']))
                metadata['schema_version'] = SCHEMA
                metadata['training'] = dict(method='masked_episode_balanced_actor_bc',
                    warm_start_sha256=WARM_SHA, phase_clock=SEMANTICS, amp_resume_compatible=False)
                payload = dict(schema=SCHEMA, model_state_dict=model.state_dict(), config=source['config'],
                    metadata=metadata, optimizer_state_dict=optimizer.state_dict(),
                    generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    updates=step, warm_start_updates=200, warm_start_sha256=WARM_SHA,
                    plan=plan, frozen_sha256=frozen, best=best,
                    observation_semantics=SEMANTICS, amp_resume_compatible=False)
                with path.open('xb') as stream:
                    torch.save(payload, stream)
                receipt = sha256(path)
                report['checkpoint_receipts'][path.name] = receipt
                print('CHECKPOINT_SAVED '+json.dumps(dict(iteration=step, path=str(path.resolve()),
                    bytes=path.stat().st_size, sha256=receipt)), flush=True)
                (args.output/'checkpoint_receipts.json').write_text(json.dumps(report['checkpoint_receipts'], indent=2)+'\n', encoding='utf-8')
            with (args.output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, loss_rad2=float(loss), gradient_norm=gradient, fit=current), allow_nan=False)+'\n')
            if step % 10 == 0:
                print('RECOVERY_BC_STEP '+json.dumps(dict(step=step, fit=current)), flush=True)
        unchanged = state_digest(model.state_dict()) == before
        if not args.train and (not unchanged or optimizer.state):
            raise ValueError('Recovery dry-run changed model/optimizer')
        report.update(status='completed' if args.train else 'prepared_not_started', best=best,
            final_fit=metrics(), model_unchanged=unchanged, frozen_exact=frozen_digest(model) == frozen,
            optimizer_state_entries=len(optimizer.state), initial_state_sha256=before)
        fit = best['fit']
        report['offline_fit_passed'] = bool(args.train and best['checkpoint'] is not None
            and fit['nominal']['raw_target_rmse_rad'] <= .06
            and fit['recovery']['raw_target_rmse_rad'] <= .06
            and fit['recovery']['raw_target_rmse_rad'] <= baseline['recovery']['raw_target_rmse_rad']*.7)
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('RECOVERY_BC_RESULT '+json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
