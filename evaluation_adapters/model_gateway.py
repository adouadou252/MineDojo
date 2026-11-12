"""Meter valid responses handed to the Agent; retry policy belongs to the Agent.

Failed or withheld responses do not spend the competition token allowance. They
may still incur real upstream charges. This development proxy is not a sandbox.
"""
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .pdf_scoring import SECONDS, TOKENS

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def usage_total(result):
    usage = result.get('usage') if isinstance(result, dict) else None
    value = usage.get('total_tokens') if isinstance(usage, dict) else None
    if type(value) is not int or value < 0:
        raise ValueError('upstream usage.total_tokens missing or invalid')
    return value


def read_result(response):
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError('upstream response too large')
    return json.loads(raw)


def upstream_id(response, result=None):
    headers = getattr(response, 'headers', None) or {}
    value = headers.get('X-Request-ID') or headers.get('Request-ID')
    if not value and isinstance(result, dict):
        value = result.get('id')
    return value[:256] if isinstance(value, str) else None


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Budget:
    def __init__(self, directory, *, clock=time.monotonic):
        self.directory = Path(directory)
        self.clock = clock
        self.lock = threading.RLock()
        self.started = None
        self.total = 0
        self.calls = 0
        self.exhausted_at = None
        self.fault = None
        self.last_error = None

    def start(self):
        with self.lock:
            self.started = self.clock()
            self.save()

    def remaining(self):
        return max(0, SECONDS - (self.clock() - self.started)) if self.started is not None else 0

    def reason(self):
        with self.lock:
            if self.fault:
                return 'accounting_error'
            if self.started is not None and self.remaining() <= 0:
                return 'wall_time'
            if self.exhausted_at is not None:
                return 'token_limit'
            return None

    def record(self, usage, request_id=None, *, gateway_request_id=None):
        with self.lock:
            if type(usage) is not int or usage < 0:
                raise ValueError('upstream usage.total_tokens missing or invalid')
            self.total += usage
            self.calls += 1
            when = self.clock()
            if self.total >= TOKENS and self.exhausted_at is None:
                self.exhausted_at = when
            with (self.directory / 'api-usage.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'call': self.calls, 'request_id': request_id,
                                      'gateway_request_id': gateway_request_id,
                                      'tokens': usage, 'total_tokens': self.total,
                                      'completed_at_monotonic': when}) + '\n')
            self.save()

    def note_error(self, kind, *, request_id, error, http_status=None, upstream_request_id=None, reported_tokens=None):
        with self.lock:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            self.last_error = {'kind': kind, 'gateway_request_id': request_id,
                               'exception': type(error).__name__,
                               'cause': type(reason).__name__, 'http_status': http_status,
                               'upstream_request_id': upstream_request_id,
                               'upstream_reported_tokens': reported_tokens,
                               'at_monotonic': self.clock()}
            with (self.directory / 'api-errors.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(self.last_error) + '\n')
            self.save()

    def fail(self, message):
        with self.lock:
            self.fault = message
            self.save()

    def save(self):
        atomic_json(self.directory / 'budget.json', self.snapshot())

    def snapshot(self):
        with self.lock:
            return {'started_monotonic': self.started, 'seconds_limit': SECONDS,
                    'tokens_limit': TOKENS, 'tokens': self.total, 'calls': self.calls,
                    'exhausted_at_monotonic': self.exhausted_at, 'fault': self.fault,
                    'last_error': self.last_error, 'metering_scope': 'delivered_responses'}


class ModelGateway:
    def __init__(self, budget, endpoint, key, model, *, opener=None):
        self.budget, self.endpoint, self.key, self.model = budget, endpoint, key, model
        self.token = secrets.token_urlsafe(32)
        self.serial = threading.Lock()
        # Keep the designated endpoint and credential; never follow redirects.
        self.open = opener or urllib.request.build_opener(NoRedirect()).open
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, payload, request_id=None):
                encoded = json.dumps(payload).encode()
                request_id = request_id or (payload.get('gateway_request_id') if isinstance(payload, dict) else None)
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(encoded)))
                    if request_id:
                        self.send_header('X-Request-ID', request_id)
                    self.end_headers()
                    self.wfile.write(encoded)
                except OSError:
                    pass  # Usage was persisted before forwarding to the Agent.

            def do_POST(self):
                supplied = self.headers.get('X-Evaluation-API-Token', '')
                if supplied != owner.token and self.headers.get('Authorization') != 'Bearer ' + owner.token:
                    return self.reply(403, {'error': 'invalid episode token'})
                if self.path != '/v1/chat/completions':
                    return self.reply(404, {'error': 'only the designated model endpoint is allowed'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 16 * 1024 * 1024:
                        raise ValueError('request size')
                    self.connection.settimeout(max(.1, min(30, owner.budget.remaining())))
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or payload.get('stream'):
                        raise ValueError('only non-streaming chat completions are supported')
                    payload['model'] = owner.model
                except (ValueError, OSError):
                    return self.reply(400, {'error': 'invalid or streaming request'})
                request_id = secrets.token_hex(16)
                status, result = owner.forward(payload, request_id=request_id)
                return self.reply(status, result, request_id)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = 'http://127.0.0.1:%d/v1/chat/completions' % self.server.server_address[1]

    def invalidate(self, error, request_id):
        message = '计量记录无法保存；本局已停止，请联系组织方。' + type(error).__name__
        try:
            self.budget.fail(message)
        except Exception:
            # fail() sets the in-memory latch before writing the broken disk.
            pass
        return 502, {'error': message, 'code': 'accounting_error', 'gateway_request_id': request_id}

    def request_error(self, error, request_id, *, kind, status=502, remote_status=None,
                      remote_id=None, reported_tokens=None):
        try:
            self.budget.note_error(kind, request_id=request_id, error=error,
                http_status=remote_status, upstream_request_id=remote_id, reported_tokens=reported_tokens)
        except Exception as storage_error:
            return self.invalidate(storage_error, request_id)
        # No upstream error body, output, credentials or arbitrary exception text
        # is forwarded. The Agent receives the original HTTP error category.
        message = '模型请求失败，本次未交付模型内容；由 Agent 决定后续处理。'
        if remote_status is not None:
            message += '上游 HTTP %s。' % remote_status
        if kind in ('wall_time', 'token_limit', 'accounting_error'):
            message = kind
        return status, {'error': message, 'code': kind, 'gateway_request_id': request_id,
                        'upstream_status': remote_status}

    def forward(self, payload, *, request_id=None):
        with self.serial:
            request_id = request_id or secrets.token_hex(16)
            reason = self.budget.reason()
            if reason or self.budget.started is None:
                return 429, {'error': reason or 'world not started', 'gateway_request_id': request_id}
            if not self.key or not self.model:
                return 503, {'error': 'configure official model and API key for LLM use'}
            status = remote_id = tokens = None
            stage = 'request'
            try:
                request = urllib.request.Request(self.endpoint,
                    data=json.dumps({**payload, 'model': self.model}).encode(),
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                             'X-Request-ID': request_id}, method='POST')
                stage = 'transport'
                # Exactly ONE upstream attempt for each Agent request. Retries
                # never reset the per-world clock or competition token balance.
                with self.open(request, timeout=self.budget.remaining()) as response:
                    status, remote_id = response.status, upstream_id(response)
                    stage = 'response'
                    result = read_result(response)
                    remote_id = upstream_id(response, result)
                    tokens = usage_total(result)
                    choices = result.get('choices')
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise ValueError('missing choices')
                    message = choices[0].get('message')
                    if not isinstance(message, dict):
                        raise ValueError('missing message')
                    content, tools = message.get('content'), message.get('tool_calls')
                    if not ((isinstance(content, str) and content.strip()) or
                            (isinstance(tools, list) and tools)):
                        raise ValueError('no model content')
            except urllib.error.HTTPError as error:
                status, remote_id = error.code, upstream_id(error)
                # Record known real usage separately; do not treat an unknown
                # charge as zero and do not charge failed replies to the contest.
                try:
                    with error:
                        body = read_result(error)
                        remote_id = upstream_id(error, body)
                        tokens = usage_total(body)
                except Exception:
                    tokens = None
                return self.request_error(error, request_id, kind='upstream_http_error',
                    status=status if 400 <= status <= 599 else 502, remote_status=status,
                    remote_id=remote_id, reported_tokens=tokens)
            except Exception as error:
                cause = error.reason if isinstance(error, urllib.error.URLError) else error
                timeout = isinstance(cause, TimeoutError)
                return self.request_error(error, request_id,
                    kind='upstream_timeout' if timeout else 'upstream_' + stage + '_error',
                    status=504 if timeout else 502, remote_status=status,
                    remote_id=remote_id, reported_tokens=tokens)
            # Never hand unmetered content to the Agent. The durable commit is
            # the handoff boundary; later Agent-side discard/disconnect does not
            # refund content that may already have been observed.
            reason = self.budget.reason()
            if reason:
                return self.request_error(RuntimeError(reason), request_id, kind=reason, status=429,
                    remote_status=status, remote_id=remote_id, reported_tokens=tokens)
            try:
                if self.budget.total + tokens > TOKENS:
                    # Withhold an over-budget reply without charging undelivered
                    # tokens, then latch the normal token limit (not a fault).
                    with self.budget.lock:
                        self.budget.exhausted_at = self.budget.clock()
                        self.budget.save()
                    return self.request_error(RuntimeError('token_limit'), request_id,
                        kind='token_limit', status=429, remote_status=status,
                        remote_id=remote_id, reported_tokens=tokens)
                self.budget.record(tokens, remote_id, gateway_request_id=request_id)
            except Exception as storage_error:
                return self.invalidate(storage_error, request_id)
            # Exact-cap replies are delivered and counted; the runner refuses
            # subsequent environment actions/calls once reason()==token_limit.
            return 200, result

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
