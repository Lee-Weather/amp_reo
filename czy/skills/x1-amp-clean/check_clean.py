#!/usr/bin/env python
"""Verify the X1 AMP clean-branch invariants.

Usage:
    python czy/skills/x1-amp-clean/check_clean.py [repo_root]

Exit code 0 when every check passes, 1 otherwise. Only the standard library is
used, so the checker runs without Isaac Gym or CUDA.
"""
import ast
import importlib
import pathlib
import sys
import types

ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[3]

EXPECTED_ENV_FILES = {
    "x1_amp_config.py",
    "x1_amp_env.py",
    "x1_dh_stand_config.py",
    "x1_dh_stand_env.py",
}
EXPECTED_TASKS = {"x1_dh_stand", "x1_amp", "x1_amp_base"}
REQUIRED_AMP_MODULES = [
    "algo/ppo/actor_critic_dh.py", "algo/ppo/dh_on_policy_runner.py", "algo/ppo/dh_ppo.py",
    "algo/ppo/rollout_storage.py", "algo/vec_env.py",
    "amp/anchored_runner.py", "amp/checkpoint.py", "amp/conservative_runner.py",
    "amp/contact_domain.py", "amp/control.py", "amp/dh_models.py",
    "amp/exp20_commands.py", "amp/exp20_rewards.py", "amp/exp23_curriculum.py",
    "amp/exp23_runner.py", "amp/exp24_curriculum.py", "amp/exp24_runner.py",
    "amp/exp25_curriculum.py", "amp/exp25_runner.py", "amp/exp26_curriculum.py",
    "amp/exp26_swing.py", "amp/exp27_balance.py", "amp/exp27_curriculum.py",
    "amp/exp28_curriculum.py", "amp/exp28_tracking.py", "amp/exp29_curriculum.py",
    "amp/exp29_phase.py", "amp/gait_curriculum.py", "amp/gait_guidance.py",
    "amp/gait_rewards.py", "amp/grounding.py", "amp/logging.py", "amp/models.py",
    "amp/motion.py", "amp/native_reference.py", "amp/nominal_startup.py",
    "amp/phase_initialization.py", "amp/phase_reference.py", "amp/phase_rewards.py",
    "amp/playback.py", "amp/reference_distance.py", "amp/replay.py",
    "amp/reward_transfer.py", "amp/runner.py", "amp/stationary.py", "amp/teacher_transfer.py",
]
REQUIRED_DATA = [
    "amp/reference_data/exp20_native_correction.json",
    "amp/reference_data/exp20_sole_hulls.json",
    "amp/reference_data/exp22_policy_state_bank.json",
]
EXPECTED_TEST_COUNT = 11

results = []


def check(ok, message):
    results.append((bool(ok), message))


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def class_bases(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [ast.unparse(b) for b in node.bases]
    return None


def task_registrations(tree):
    tasks = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "task_registry"
                and node.args and isinstance(node.args[0], ast.Constant)):
            tasks.append(node.args[0].value)
    return tasks


def scan_exp_imports(paths):
    offenders = []
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        tree = parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                if "x1_amp_exp" in module:
                    offenders.append("{}:{} {}".format(path, node.lineno, module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "x1_amp_exp" in alias.name:
                        offenders.append("{}:{} {}".format(path, node.lineno, alias.name))
    return offenders


def resolve_configs():
    sys.path.insert(0, str(ROOT))
    import humanoid  # noqa: F401  (light module: only defines paths)
    package = types.ModuleType("humanoid.envs")
    package.__path__ = [str(ROOT / "humanoid" / "envs")]
    sys.modules["humanoid.envs"] = package
    module = importlib.import_module("humanoid.envs.x1.x1_amp_config")
    return module


def main():
    env_dir = ROOT / "humanoid" / "envs" / "x1"
    env_files = {p.name for p in env_dir.glob("*.py")}
    check(env_files == EXPECTED_ENV_FILES,
          "envs/x1 files == {}".format(sorted(EXPECTED_ENV_FILES)))
    if env_files != EXPECTED_ENV_FILES:
        check(False, "unexpected env files: +{} -{}".format(
            sorted(env_files - EXPECTED_ENV_FILES), sorted(EXPECTED_ENV_FILES - env_files)))

    init_path = ROOT / "humanoid" / "envs" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    tasks = task_registrations(parse(init_path))
    check(set(tasks) == EXPECTED_TASKS and len(tasks) == 3,
          "registry tasks == {}".format(sorted(EXPECTED_TASKS)))
    if set(tasks) != EXPECTED_TASKS:
        check(False, "unexpected tasks: {}".format(tasks))
    check(not scan_exp_imports([init_path]), "registry imports no x1_amp_exp modules")

    missing = [m for m in REQUIRED_AMP_MODULES if not (ROOT / "humanoid" / m).exists()]
    check(not missing, "required humanoid/amp modules present")
    if missing:
        check(False, "missing amp modules: {}".format(missing))

    missing = [d for d in REQUIRED_DATA if not (ROOT / "humanoid" / d).exists()]
    check(not missing, "humanoid/amp/reference_data present")
    if missing:
        check(False, "missing reference data: {}".format(missing))

    scan_paths = list((ROOT / "humanoid").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    offenders = scan_exp_imports(scan_paths)
    check(not offenders, "no x1_amp_exp module imports in humanoid/scripts/tests")
    for offender in offenders[:10]:
        check(False, "exp import: " + offender)

    env_tree = parse(env_dir / "x1_amp_env.py")
    check(class_bases(env_tree, "X1AMPFinalEnv") == ["LeggedRobot"],
          "X1AMPFinalEnv inherits LeggedRobot directly")
    check(class_bases(env_tree, "X1AMPEnv") == ["LeggedRobot"],
          "X1AMPEnv inherits LeggedRobot directly")

    cfg_tree = parse(env_dir / "x1_amp_config.py")
    check(class_bases(cfg_tree, "X1AMPFinalCfg") == ["LeggedRobotCfg"],
          "X1AMPFinalCfg inherits LeggedRobotCfg directly")
    check(class_bases(cfg_tree, "X1AMPCfg") == ["LeggedRobotCfg"],
          "X1AMPCfg inherits LeggedRobotCfg directly")

    try:
        module = resolve_configs()
        final = module.X1AMPFinalCfg()
        base = module.X1AMPCfg()
        final_ppo = module.X1AMPFinalCfgPPO()
        base_ppo = module.X1AMPCfgPPO()
        check(final.env.num_observations == 6468 and final.env.num_privileged_obs == 101,
              "x1_amp observation dimension 6468 / 101")
        check(final.env.frame_stack == 66 and final.env.num_single_obs == 98,
              "x1_amp frame stack 66 x 98")
        check(final.amp.history_steps == 3, "x1_amp AMP history 3 frames")
        check(abs(final.commands.native_speed - 0.5953696966) < 1e-9,
              "x1_amp native speed 0.5953696966")
        check(abs(final.amp.nominal_startup_fraction - 0.25) < 1e-9,
              "x1_amp nominal startup fraction 0.25")
        check(bool(final.domain_rand.randomize_contact_offset),
              "x1_amp contact-offset randomisation enabled")
        check(final_ppo.runner_class_name == "Exp25AMPOnPolicyRunner",
              "x1_amp runner Exp25AMPOnPolicyRunner")
        check(getattr(final_ppo.runner, "policy_class_name", None) == "ActorCriticDHAMP",
              "x1_amp policy ActorCriticDHAMP")
        check(final_ppo.runner.experiment_name == "x1_amp",
              "x1_amp experiment name x1_amp")
        check(base.env.num_observations == 480 and base.env.num_privileged_obs == 99,
              "x1_amp_base observation dimension 480 / 99")
        check(base_ppo.runner_class_name == "AMPOnPolicyRunner",
              "x1_amp_base runner AMPOnPolicyRunner")
    except Exception as error:  # pragma: no cover - reported as a failed check
        check(False, "config resolution failed: {!r}".format(error))

    tests = sorted((ROOT / "tests").glob("test_*.py"))
    check(len(tests) == EXPECTED_TEST_COUNT,
          "tests == {} files".format(EXPECTED_TEST_COUNT))
    if len(tests) != EXPECTED_TEST_COUNT:
        check(False, "unexpected test count: {}".format(len(tests)))

    width = max(len(message) for _, message in results)
    failed = 0
    for ok, message in results:
        print(("PASS  " if ok else "FAIL  ") + message.ljust(width))
        failed += 0 if ok else 1
    print("\n{}/{} checks passed".format(len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())