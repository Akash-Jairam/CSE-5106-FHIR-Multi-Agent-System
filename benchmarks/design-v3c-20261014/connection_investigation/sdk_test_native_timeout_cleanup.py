"""No network or provider calls: real HTTPX2/HTTPCORE2 pool with a fake backend."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
DEPS = ROOT / '.local-setup/phase2-dependencies'
sys.path[:0] = [str(DEPS), str(ROOT / '.venv/lib/python3.12/site-packages')]

import httpx2
import httpcore2
from httpcore2._backends.mock import MockBackend, MockStream

class FailedRead(MockStream):
    def __init__(self, body_stage=False):
        super().__init__([])
        self.body_stage = body_stage
        self.reads = 0

    def read(self, max_bytes, timeout=None):
        self.reads += 1
        if self.body_stage and self.reads == 1:
            return b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n'
        raise httpcore2.ReadTimeout('synthetic read timeout')

class SequenceBackend(MockBackend):
    def __init__(self, failure):
        super().__init__([])
        self.failure = failure
        self.calls = 0
        self.failed_stream = None

    def connect_tcp(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            if self.failure == 'connect_timeout':
                raise httpcore2.ConnectTimeout('synthetic connect timeout')
            if self.failure == 'connect_error':
                raise httpcore2.ConnectError('synthetic DNS/socket/TLS wrapper')
            self.failed_stream = FailedRead(body_stage=self.failure == 'body_read_timeout')
            return self.failed_stream
        return MockStream([b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK'])

def scenario(failure):
    transport = httpx2.HTTPTransport()
    backend = SequenceBackend(failure)
    transport._pool._network_backend = backend
    client = httpx2.Client(transport=transport, timeout=60.0)
    expected = httpx2.ConnectTimeout if failure == 'connect_timeout' else httpx2.ConnectError if failure == 'connect_error' else httpx2.ReadTimeout
    try:
        client.get('http://offline.invalid/test')
    except expected as exc:
        first_type = type(exc).__module__ + '.' + type(exc).__name__
    else:
        raise AssertionError('synthetic failure was not raised')
    pool = transport._pool
    after_failure = {'client_closed': client.is_closed, 'connections': len(pool.connections), 'requests': len(pool._requests), 'failed_stream_closed': None if backend.failed_stream is None else backend.failed_stream._closed}
    assert not client.is_closed
    assert len(pool.connections) == 0
    assert len(pool._requests) == 0
    if backend.failed_stream is not None:
        assert backend.failed_stream._closed
    response = client.get('http://offline.invalid/test')
    assert response.status_code == 200 and response.text == 'OK'
    assert backend.calls == 2
    assert len(pool._requests) == 0
    assert not client.is_closed
    after_success = {'client_closed': client.is_closed, 'connections': len(pool.connections), 'requests': len(pool._requests), 'backend_connects': backend.calls}
    client.close()
    assert client.is_closed
    return {'failure': failure, 'exception': first_type, 'after_failure': after_failure, 'after_success': after_success}

def provenance():
    entries = json.loads((Path(__file__).parent / 'public_sources.json').read_text())
    mapping = {'sdkbase': ROOT / '.venv/lib/python3.12/site-packages/openai/_base_client.py', 'sdkcompat': ROOT / '.venv/lib/python3.12/site-packages/openai/_httpx2.py', 'nativeclient': DEPS / 'httpx2/_client.py', 'nativetransport': DEPS / 'httpx2/_transports/default.py', 'pool': DEPS / 'httpcore2/_sync/connection_pool.py', 'connection': DEPS / 'httpcore2/_sync/connection.py', 'http11': DEPS / 'httpcore2/_sync/http11.py', 'backend': DEPS / 'httpcore2/_backends/sync.py'}
    for entry in entries:
        data = mapping[entry['name']].read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        entry.update(local_path=str(mapping[entry['name']]), sha256=hashlib.sha256(data).hexdigest(), local_git_blob_sha=blob, matches_official_tag=blob == entry['git_blob_sha'])
        assert entry['matches_official_tag']
    return entries

if __name__ == '__main__':
    output = {'no_network': True, 'scope': 'synthetic ordinary sync HTTP1 failures, not reproduction of benchmark collapse', 'versions': {name: importlib.metadata.version(name) for name in ('openai', 'httpx2', 'httpcore2')}, 'scenarios': [scenario(name) for name in ('connect_timeout', 'connect_error', 'header_read_timeout', 'body_read_timeout')], 'provenance': provenance()}
    (Path(__file__).parent / 'native_timeout_cleanup_results.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
