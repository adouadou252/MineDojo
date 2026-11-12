"""Build a root-layout Agent ZIP without including runs, credentials or venvs."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('offline', 'model'), default='offline')
    parser.add_argument('--output', type=Path, default=Path('agent-submission.zip'))
    parser.add_argument('--extra', type=Path, nargs='*', default=[], help='Initial offline memory/assets relative to this directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = {name: root/name for name in ('config.json', 'model_api.py', 'minecraft_actions.py', 'action_items.json', 'technical-route.md')}
    files['agent.py'] = root/('model_agent.py' if args.policy == 'model' else 'agent.py')
    for extra in args.extra:
        path = (root/extra).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            parser.error('--extra must stay inside the package directory')
        for file in ([path] if path.is_file() else path.rglob('*')):
            if file.is_file() and not file.is_symlink() and '__pycache__' not in file.parts:
                name = file.relative_to(root).as_posix()
                if name in files:
                    parser.error('extra asset conflicts with a required file')
                files[name] = file
    if len(files) > 256 or sum(p.stat().st_size for p in files.values()) > 64*1024*1024:
        parser.error('Agent ZIP limit: 256 files, 64 MiB extracted')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if name == 'config.json':
                config = json.loads(path.read_text(encoding='utf-8'))
                # Inject development/evaluation credentials at runtime, never ship a permanent key.
                config['model']['api_key'] = ''
                archive.writestr(name, json.dumps(config, ensure_ascii=False))
            else:
                archive.write(path, name)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
