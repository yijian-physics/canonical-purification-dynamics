#!/usr/bin/env python3
"""Publish one immutable benchmark directory without losing concurrent commits.

Run from a clean checkout. All raw files under --results are committed, including
failure logs and partial runs. A normal push is retried after fetch/rebase; no
force push, reset, clean, deletion, or overwrite of an existing run is used.
On failure the working directory and local result commit remain available for
artifact upload. GitHub's workflow uploads the raw files before calling this.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time


class PublicationError(RuntimeError):
    pass


def git(*args, check=True):
    result = subprocess.run(['git', *args], text=True, capture_output=True)
    if check and result.returncode:
        raise PublicationError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result


def tree_at(revision, path):
    result = git('rev-parse', '--verify', f'{revision}:{path}', check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def publish(results, branch, attempts=5, retry_seconds=2):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', branch):
        raise PublicationError('Invalid publication branch')
    git('check-ref-format', f'refs/heads/{branch}')
    root = Path(git('rev-parse', '--show-toplevel').stdout.strip()).resolve()
    os.chdir(root)
    raw_path = Path(results)
    if raw_path.is_absolute() or '..' in raw_path.parts or raw_path == Path('.'):
        raise PublicationError('Results must be a nonempty repository-relative directory')
    path = root / raw_path
    if not path.is_dir() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise PublicationError('Results directory is missing or escapes the repository')
    if '.git' in raw_path.parts or any(p.is_symlink() for p in path.rglob('*')):
        raise PublicationError('Results may not contain symlinks or Git metadata')
    relative = raw_path.as_posix()
    if not any(p.is_file() for p in path.rglob('*')):
        raise PublicationError('Results directory has no files to preserve')
    if git('diff', '--quiet', check=False).returncode:
        raise PublicationError('Tracked files changed outside a fresh immutable run; preserve and review them')
    if git('diff', '--cached', '--quiet', check=False).returncode:
        raise PublicationError('Refusing to include pre-existing staged changes')
    # Existing tracked runs may be retried only byte-for-byte. Untracked files
    # under an old run would mutate its historical contents, so refuse them.
    old_tree = tree_at('HEAD', relative)
    untracked = git('ls-files', '--others', '--exclude-standard', '--', relative).stdout
    ignored = git('ls-files', '--others', '--ignored', '--exclude-standard', '--', relative).stdout
    if old_tree and (untracked or ignored):
        raise PublicationError('Refusing to append files to an existing immutable run')
    if not old_tree:
        git('add', '--force', '--', relative)
        if git('diff', '--cached', '--quiet', check=False).returncode == 0:
            raise PublicationError('No benchmark files were staged')
        git('-c', 'user.name=github-actions[bot]', '-c',
            'user.email=41898282+github-actions[bot]@users.noreply.github.com',
            'commit', '-m', f'Record immutable benchmark run {raw_path.name}', '--', relative)
    local_tree = tree_at('HEAD', relative)
    for attempt in range(1, attempts + 1):
        print(f'Publication attempt {attempt}/{attempts}', flush=True)
        fetched = git('fetch', '--no-tags', 'origin', f'refs/heads/{branch}', check=False)
        if fetched.returncode:
            print(fetched.stderr, file=sys.stderr, flush=True)
        else:
            remote_tree = tree_at('FETCH_HEAD', relative)
            if remote_tree:
                if remote_tree == local_tree:
                    print('Identical immutable run is already published.', flush=True)
                    return
                raise PublicationError('Remote already contains different files at this run path; refusing overwrite')
            rebased = git('-c', 'user.name=github-actions[bot]', '-c',
                          'user.email=41898282+github-actions[bot]@users.noreply.github.com',
                          'rebase', 'FETCH_HEAD', check=False)
            if rebased.returncode:
                # Leave conflicts and the local commit for inspection; do not
                # abort/reset away evidence. Artifact upload is independent.
                raise PublicationError('Rebase failed; raw results and local commit preserved. ' + rebased.stderr.strip())
            if tree_at('HEAD', relative) != local_tree:
                raise PublicationError('Rebase changed the benchmark directory unexpectedly')
            pushed = git('push', 'origin', f'HEAD:refs/heads/{branch}', check=False)
            if pushed.returncode == 0:
                print(pushed.stdout + pushed.stderr, end='', flush=True)
                print('Published immutable benchmark files with a normal push.', flush=True)
                return
            print(pushed.stderr, file=sys.stderr, flush=True)
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise PublicationError('Publication retries exhausted; local result commit and all raw files are preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--retry-seconds', type=float, default=2)
    args = parser.parse_args()
    if args.attempts < 1 or args.retry_seconds < 0:
        parser.error('attempts must be positive and retry-seconds nonnegative')
    try:
        publish(args.results, args.branch, args.attempts, args.retry_seconds)
    except (PublicationError, OSError) as exc:
        print(f'Publication failed: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
