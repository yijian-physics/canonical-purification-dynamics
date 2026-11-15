"""Offline integration tests against real temporary Git repositories.

Run: python -m unittest discover -s .github/scripts -p 'test_publish*.py' -v
"""
from pathlib import Path
import os
import shlex
import subprocess
import sys
import tempfile
import unittest

PUBLISH = Path(__file__).with_name('publish_benchmark_results.py').resolve()
BRANCH = 'research/factor-vs-witness-scaling'
RUN = 'factor_tdvp_accelerated/size_scaling_results/actions-123-attempt-1'


class PublicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='benchmark-publication-')
        self.root = Path(self.temp.name)
        self.remote, self.worker, self.racer = [self.root / name for name in ('remote.git', 'worker', 'racer')]
        # Keep the test isolated from machine-wide Git configuration/hooks.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.git(self.root, 'init', '--bare', str(self.remote))
        self.git(self.root, 'clone', str(self.remote), str(self.worker))
        self.git(self.worker, 'checkout', '-b', BRANCH)
        self.identity(self.worker)
        (self.worker / 'source.py').write_text('original implementation\n')
        self.git(self.worker, 'add', 'source.py')
        self.git(self.worker, 'commit', '-m', 'initial source')
        self.git(self.worker, 'push', '-u', 'origin', BRANCH)
        self.git(self.root, 'clone', '--branch', BRANCH, str(self.remote), str(self.racer))
        self.identity(self.racer)
        # Actions does not guarantee a configured identity for rebase.
        self.git(self.worker, 'config', '--unset', 'user.name')
        self.git(self.worker, 'config', '--unset', 'user.email')
        self.results = self.worker / RUN
        self.results.mkdir(parents=True)
        (self.results / 'measurements.json').write_text('[{"status":"timeout"}]\n')
        (self.results / 'case-001').mkdir()
        (self.results / 'case-001' / 'stderr.log').write_text('retained numerical failure\n')
        (self.results / 'case-001' / '.hidden-diagnostic').write_text('retained hidden output\n')

    def tearDown(self):
        self.temp.cleanup()

    def git(self, cwd, *args, check=True):
        return subprocess.run(['git', *args], cwd=cwd, env=self.env,
                              text=True, capture_output=True, check=check)

    def identity(self, cwd):
        self.git(cwd, 'config', 'user.name', 'Publication test')
        self.git(cwd, 'config', 'user.email', 'test@example.invalid')

    def publish(self, attempts=3):
        return subprocess.run([sys.executable, str(PUBLISH), '--results', RUN,
                               '--branch', BRANCH, '--attempts', str(attempts),
                               '--retry-seconds', '0'], cwd=self.worker, env=self.env,
                              text=True, capture_output=True)

    def remote_file(self, path):
        return self.git(self.remote, 'show', f'refs/heads/{BRANCH}:{path}').stdout

    def test_remote_advance_during_push_rebases_without_losing_logs(self):
        # The first push's pre-push hook advances origin AFTER the publisher's
        # fetch/rebase, reproducing the actual race rather than a stale clone.
        racer_script = self.root / 'advance_remote.py'
        racer_script.write_text('''import os, pathlib, subprocess, sys
repo=pathlib.Path(sys.argv[1])
env={k:v for k,v in os.environ.items() if not k.startswith("GIT_")}
env.update(GIT_CONFIG_NOSYSTEM="1",GIT_CONFIG_GLOBAL=os.devnull)
(repo/"source.py").write_text("concurrent source improvement\\n")
for args in (("add","source.py"),("commit","-m","concurrent source update"),("push","origin",sys.argv[2])):
 subprocess.run(["git",*args],cwd=repo,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
''')
        marker = self.root / 'race-fired'
        hook = self.worker / '.git/hooks/pre-push'
        hook.write_text('#!/bin/sh\nset -eu\n' +
                        f'if [ ! -f {shlex.quote(str(marker))} ]; then\n' +
                        f'  touch {shlex.quote(str(marker))}\n' +
                        '  ' + ' '.join(map(shlex.quote, (sys.executable, str(racer_script), str(self.racer), BRANCH))) + '\nfi\n')
        hook.chmod(0o755)
        outcome = self.publish()
        self.assertEqual(outcome.returncode, 0, outcome.stdout + outcome.stderr)
        self.assertIn('Publication attempt 2/', outcome.stdout)
        self.assertEqual(self.remote_file('source.py'), 'concurrent source improvement\n')
        self.assertEqual(self.remote_file(RUN + '/case-001/stderr.log'), 'retained numerical failure\n')
        self.assertEqual(self.remote_file(RUN + '/case-001/.hidden-diagnostic'), 'retained hidden output\n')
        self.assertIn('concurrent source update', self.git(self.remote, 'log', f'refs/heads/{BRANCH}', '--format=%s').stdout)
        first_head = self.git(self.remote, 'rev-parse', f'refs/heads/{BRANCH}').stdout
        # Network ambiguity or a manual retry cannot duplicate a published run.
        again = self.publish()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn('already published', again.stdout)
        self.assertEqual(first_head, self.git(self.remote, 'rev-parse', f'refs/heads/{BRANCH}').stdout)

    def test_existing_remote_run_collision_fails_without_overwrite(self):
        other = self.racer / RUN
        other.mkdir(parents=True)
        (other / 'measurements.json').write_text('earlier immutable result\n')
        self.git(self.racer, 'add', RUN)
        self.git(self.racer, 'commit', '-m', 'other publication')
        self.git(self.racer, 'push', 'origin', BRANCH)
        outcome = self.publish()
        self.assertNotEqual(outcome.returncode, 0)
        self.assertIn('refusing overwrite', outcome.stderr)
        self.assertEqual(self.remote_file(RUN + '/measurements.json'), 'earlier immutable result\n')
        self.assertEqual((self.results / 'case-001/stderr.log').read_text(), 'retained numerical failure\n')
        self.assertIn('Record immutable benchmark run', self.git(self.worker, 'log', '-1', '--format=%s').stdout)

    def test_failed_push_preserves_local_commit_and_partial_raw_files(self):
        hook = self.remote / 'hooks/pre-receive'
        hook.write_text('#!/bin/sh\necho publication-disabled >&2\nexit 1\n')
        hook.chmod(0o755)
        outcome = self.publish(attempts=2)
        self.assertNotEqual(outcome.returncode, 0)
        self.assertIn('retries exhausted', outcome.stderr)
        self.assertTrue((self.results / 'case-001/stderr.log').is_file())
        self.assertIn('Record immutable benchmark run', self.git(self.worker, 'log', '-1', '--format=%s').stdout)
        self.assertEqual(self.remote_file('source.py'), 'original implementation\n')

    def test_unrelated_tracked_edits_are_not_published(self):
        (self.worker / 'source.py').write_text('unreviewed edit\n')
        outcome = self.publish()
        self.assertNotEqual(outcome.returncode, 0)
        self.assertIn('Tracked files changed', outcome.stderr)
        self.assertEqual((self.worker / 'source.py').read_text(), 'unreviewed edit\n')
        self.assertEqual(self.remote_file('source.py'), 'original implementation\n')


if __name__ == '__main__':
    unittest.main()
