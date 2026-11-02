"""A completed model must replay with its own committed code, not later edits."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("replay_archive", Path(__file__).resolve().parents[1]/"scripts/amp_tools/replay_archive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReplayRevisionTests(unittest.TestCase):
    def test_idle_desktop_does_not_hide_a_replay_or_training_process(self):
        query = "10665, /usr/NX/bin/nxnode.bin, 292\n133659, python, 5800\n"
        active, desktop = module.gpu_workloads(query)
        self.assertEqual([p["pid"] for p in active], [133659])
        self.assertEqual([p["pid"] for p in desktop], [10665])
        self.assertTrue(module.gpu_workloads("10665, /usr/NX/bin/nxnode.bin, 4096\n")[0])
        self.assertEqual(module.gpu_workloads(""), ([], []))
        with self.assertRaises(ValueError):
            module.gpu_workloads("unknown GPU query format")

    def test_training_commit_ignores_later_local_code_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"humanoid/scripts").mkdir(parents=True)
            (root/"humanoid/scripts/play_amp.py").write_text("version = 'trained'\n")
            (root/"setup.py").write_text("name = 'x1'\n")
            (root/"credentials.json").write_text('{"private":true}')
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=ReplayTest", "-c", "user.email=replay@example.invalid", "commit", "-qm", "training")
            commit = git("rev-parse", "HEAD")
            (root/"humanoid/scripts/play_amp.py").write_text("version = 'later'\n")
            code, verified = module.source_files(commit, root)
            self.assertEqual(verified, commit)
            self.assertEqual(set(code), {"humanoid/scripts/play_amp.py", "setup.py"})
            self.assertEqual(code["humanoid/scripts/play_amp.py"], b"version = 'trained'\n")
            current, revision = module.source_files(None, root)
            self.assertIsNone(revision)
            self.assertIn(b"later", current["humanoid/scripts/play_amp.py"])

    def test_revision_must_be_an_explicit_full_sha(self):
        for revision in ("HEAD", "main", "../../secret", "a"*39, "a"*40+";echo unsafe"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                module.source_files(revision)

    def test_runtime_reference_json_is_packaged_without_unrelated_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'humanoid/amp/reference_data').mkdir(parents=True)
            (root/'humanoid/scripts').mkdir(parents=True)
            (root/'humanoid/scripts/play_amp.py').write_text('version=1\n')
            (root/'setup.py').write_text('name="test"\n')
            reference = root/'humanoid/amp/reference_data/cycle.json'
            reference.write_text('{"reference":true}\n')
            (root/'credentials.json').write_text('{"private":true}\n')
            code,_ = module.source_files(None,root)
            self.assertIn('humanoid/amp/reference_data/cycle.json',code)
            self.assertNotIn('credentials.json',code)


if __name__ == "__main__":
    unittest.main()
