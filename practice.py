"""Run tutorial examples and independent full PDF worlds locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from evaluation_adapters.model_gateway import atomic_json
from evaluation_adapters.pdf_runner import validate_scenarios
from evaluation_adapters.pdf_scoring import aggregate


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return None


def stop_tree(process):
    # Dedicated process group: also stop stuck Agent, Java and video encoder.
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def settle(directory, scene, forced=None):
    complete = read_json(directory / 'outcome.json')
    checkpoint = read_json(directory / 'checkpoint.json')
    meter = read_json(directory / 'budget.json') or {}
    if complete:
        return complete
    result = checkpoint or {'scenario': scene, 'scored': scene['mode'] == 'full',
                            'scoring': None, 'metrics': {}, 'objectives': []}
    reason = forced
    if not reason and meter.get('exhausted_at_monotonic') is not None:
        reason = 'token_limit'
    if (not reason and meter.get('started_monotonic') is not None and
            time.monotonic() - meter['started_monotonic'] >= 3600):
        reason = 'wall_time'
    # A fully committed terminal step wins over a later recorder/cleanup timeout.
    if checkpoint and checkpoint['scoring'] and checkpoint['scoring']['first_acquired']['iron_pickaxe'] is not None:
        reason = 'iron_pickaxe'
    if meter.get('fault'):
        reason = 'accounting_error'
    result['status'] = 'completed' if checkpoint and reason in ('wall_time', 'token_limit', 'iron_pickaxe') else 'error'
    result['termination'] = reason or 'runtime_error'
    result['api'] = meter
    if reason == 'wall_time':
        result['metrics']['execution_seconds'] = 3600
    atomic_json(directory / 'outcome.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='ZIP with agent.py at its root')
    parser.add_argument('--scenes', type=Path, default=Path(__file__).with_name('practice-scenes.json'))
    parser.add_argument('--output', type=Path, default=Path('runs'))
    parser.add_argument('--only', help='Run one example by id, e.g. woodcraft / stone-tools / iron-age')
    parser.add_argument('--no-video', action='store_true', help='Local debugging only; record JSONL without video')
    args = parser.parse_args()
    scenes = json.loads(args.scenes.read_text(encoding='utf-8-sig'))
    validate_scenarios(scenes)
    if args.only:
        scenes = [s for s in scenes if s['id'] == args.only]
        if not scenes:
            parser.error('unknown --only id')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='world-', dir=args.output.resolve()))
    # Immutable input snapshot shared by value, not a mutable directory shared by Agents.
    snapshot = root / 'input.zip'
    snapshot.write_bytes(args.package.read_bytes())
    atomic_json(root / 'manifest.json', {'package_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                       'scenes': scenes, 'rules': 'PDF-2026', 'kind': 'local-practice'})
    episodes = []
    for index, scene in enumerate(scenes, 1):
        directory = root / 'evidence' / str(index)
        directory.mkdir(parents=True)
        atomic_json(directory / 'scene.json', scene)
        print(f"[{index}/{len(scenes)}] {scene['label']} ({scene['mode']})", flush=True)
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent))
        command = [sys.executable, '-m', 'evaluation_adapters.practice_session',
                   '--package', str(snapshot), '--directory', str(directory)]
        if args.no_video:
            command.append('--no-video')
        with (directory / 'session.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=directory, env=environment,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name != 'nt',
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            launch = time.monotonic()
            forced = None
            complete_seen = None
            try:
                while process.poll() is None:
                    meter = read_json(directory / 'budget.json')
                    complete = read_json(directory / 'outcome.json')
                    if complete:
                        # Scoring is finished. Allow bounded replay/cleanup without changing the score.
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                        if complete_seen is not None and time.monotonic() - complete_seen > 30:
                            stop_tree(process)
                            break
                        if complete_seen is None:
                            complete_seen = time.monotonic()
                    elif meter and meter.get('started_monotonic') is not None:
                        if meter.get('fault'):
                            forced = 'accounting_error'
                        elif meter.get('exhausted_at_monotonic') is not None:
                            forced = 'token_limit'
                        elif time.monotonic() - meter['started_monotonic'] >= 3600:
                            forced = 'wall_time'
                    elif time.monotonic() - launch > 600:
                        forced = 'initialization_error'
                    if forced:
                        stop_tree(process)
                        break
                    time.sleep(.05)
            except BaseException:
                stop_tree(process)
                raise
            finally:
                if process.poll() is None:
                    stop_tree(process)
                elif os.name != 'nt':
                    # Clean surviving descendants even when the session already exited.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        result = settle(directory, scene, forced)
        print(f"  {result['status']}: {result['termination']}", flush=True)
        episodes.append(result)
        atomic_json(root / 'partial-results.json', {'episodes': episodes})
    document = {'episodes': episodes, 'summary': aggregate(episodes)}
    atomic_json(root / 'result.json', document)
    from practice_review import render
    render(root)
    print(json.dumps(document['summary'], ensure_ascii=False), flush=True)
    print(f'Results: {root}', flush=True)
    if any(e['status'] == 'error' for e in episodes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
