"""Internal entry point. One process tree per seed; no cross-world Agent state."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from .agent_config import package_config, runtime_package
from .minecraft_protocol import JsonChannel
from .model_gateway import Budget, ModelGateway, atomic_json
from .pdf_runner import create_minedojo_environment, run_episode


class FreshAgentChannel:
    def __init__(self, package, directory, gateway, model):
        self.directory = Path(directory)
        self.package, self.gateway, self.model = package, gateway, model
        self.socket = self.process = None

    def start(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            listener.settimeout(max(.1, self.gateway.budget.remaining()))
            runtime = self.directory / 'runtime-package.zip'
            runtime_package(self.package, runtime,
                            {'name': self.model or 'offline', 'api_url': self.gateway.url,
                             'api_key': self.gateway.token})
            environment = dict(os.environ)
            for name in ('UESTC_MODEL_API_KEY', 'UESTC_MODEL_API_URL', 'EVALUATION_API_TOKEN', 'EVALUATION_API_URL'):
                environment.pop(name, None)
            environment.update(EVALUATION_API_URL=self.gateway.url, EVALUATION_API_TOKEN=self.gateway.token,
                               PYTHONPATH=str(Path(__file__).resolve().parent.parent))
            self.log = (self.directory / 'agent.log').open('wb')
            self.process = subprocess.Popen([sys.executable, '-m', 'evaluation_adapters.practice_session',
                    '--agent', '--package', str(runtime), '--directory', str(self.directory),
                    '--port', str(listener.getsockname()[1])], cwd=self.directory,
                    env=environment, stdout=self.log, stderr=subprocess.STDOUT)
            self.socket = listener.accept()[0]
            self.socket.settimeout(max(.1, self.gateway.budget.remaining()))
            self.channel = JsonChannel(self.socket)

    def send(self, message):
        if self.socket is None:
            self.start()
        self.socket.settimeout(max(.1, self.gateway.budget.remaining()))
        self.channel.send(message)

    def receive(self):
        self.socket.settimeout(max(.1, self.gateway.budget.remaining()))
        return self.channel.receive()

    def close(self):
        if self.socket:
            self.socket.close()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--agent', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if args.agent:
        from .minecraft_contestant import load_agent, run
        with socket.create_connection(('127.0.0.1', args.port), timeout=3600) as connection:
            connection.settimeout(3600)
            agent = load_agent(args.package, args.directory / 'agent')
            run(agent, JsonChannel(connection))
        return
    import zipfile
    scene = json.loads((args.directory / 'scene.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(args.package) as archive:
        config = package_config(archive)
    settings = config['model']
    endpoint = os.environ.get('UESTC_MODEL_API_URL') or settings['api_url'] or 'https://api.uestcai.top/v1/chat/completions'
    key = os.environ.get('UESTC_MODEL_API_KEY') or settings['api_key']
    model = os.environ.get('UESTC_MODEL_NAME') or settings['name']
    budget = Budget(args.directory)
    env = None
    try:
        with ModelGateway(budget, endpoint, key, model) as gateway:
            channel = FreshAgentChannel(args.package, args.directory, gateway, model)
            try:
                env = create_minedojo_environment(scene)
                run_episode(channel, env, scene, budget, args.directory, record=not args.no_video)
            finally:
                channel.close()
    except BaseException:
        (args.directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        atomic_json(args.directory / 'failure.json', {'error': '环境或 Agent 执行失败，详见 error.txt'})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
