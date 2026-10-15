"""Sequential, per-case cache copies for the shared-primary mechanism study.

This module does not import the provider, tools, credentials, or patient data.
Existing tool-cache serialization is left unchanged. Base files are hashed but
never unpickled here; each followup gets independent physical copies.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import threading

_ACTIVE = threading.Lock()


def _inventory(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_symlink() or not path.is_file() or not path.name.endswith('.cache'):
            raise ValueError('Study cache contains an unexpected or linked entry')
        result[path.name] = {'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


@contextmanager
def activated_cache(directory, cache_module=None):
    """Switch globals only within one sequential worker; restore even on error."""
    if cache_module is None:
        import tools.cache as cache_module
    if not _ACTIVE.acquire(blocking=False):
        raise RuntimeError('Study cache contexts must be sequential and nonnested')
    previous = (cache_module.CACHE_DIR, cache_module.CACHE_ENABLED)
    old_load = getattr(cache_module, 'load_from_cache', None)
    counters = {'cache_observer_available': callable(old_load), 'load_calls': 0,
                'hits': 0, 'misses': 0}
    if callable(old_load):
        def counted_load(*args, **kwargs):
            value = old_load(*args, **kwargs)
            counters['load_calls'] += 1
            hit = value is not None and not (isinstance(value, dict) and 'error' in value)
            counters['hits' if hit else 'misses'] += 1
            return value
        cache_module.load_from_cache = counted_load
    try:
        cache_module.CACHE_DIR = str(Path(directory).resolve())
        cache_module.CACHE_ENABLED = True
        yield counters
    finally:
        cache_module.CACHE_DIR, cache_module.CACHE_ENABLED = previous
        if callable(old_load):
            cache_module.load_from_cache = old_load
        _ACTIVE.release()


class CaseCache:
    """Cold A cache, frozen post-A base, independent W/R/C writable copies."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.primary = self.directory / 'primary'
        self.primary.mkdir()
        self.base = self.directory / 'post_A_base'
        self.frozen_inventory = None
        self._forked = set()

    def freeze(self):
        if self.frozen_inventory is not None:
            raise ValueError('Post-A cache was already frozen')
        self.frozen_inventory = _inventory(self.primary)
        self.primary.rename(self.base)
        return self.proof()

    def verify_base(self):
        if self.frozen_inventory is None or _inventory(self.base) != self.frozen_inventory:
            raise ValueError('Immutable post-A cache base changed')

    def proof(self):
        self.verify_base()
        data = json.dumps(self.frozen_inventory, sort_keys=True, separators=(',', ':')).encode()
        return {'policy': 'cold_private_A_then_identical_physical_copies_no_hardlinks',
                'files': len(self.frozen_inventory),
                'bytes': sum(value['bytes'] for value in self.frozen_inventory.values()),
                'inventory_sha256': hashlib.sha256(data).hexdigest()}

    def fork(self, mode):
        if mode not in ('W', 'R', 'C') or mode in self._forked:
            raise ValueError('Each predeclared followup cache may be created once')
        self.verify_base()
        target = self.directory / mode
        target.mkdir(exist_ok=False)
        for name in self.frozen_inventory:
            source, destination = self.base / name, target / name
            shutil.copyfile(source, destination)
            if source.stat().st_ino == destination.stat().st_ino:
                raise ValueError('Cache fork unexpectedly aliases base inode')
        if _inventory(target) != self.frozen_inventory:
            raise ValueError('Cache fork differs from the immutable post-A base')
        self._forked.add(mode)
        return target

    def after_followup(self, mode):
        self.verify_base()
        if mode not in self._forked:
            raise ValueError('Followup cache was never created')
        current = _inventory(self.directory / mode)
        return {**self.proof(), 'followup': mode,
                'overlay_files_after': len(current),
                'new_overlay_files': len(set(current) - set(self.frozen_inventory)),
                'base_unchanged': True}


class PhysicalSourceObserver:
    """Observe existing pagination counters without issuing or changing a query.

    The injected class seam keeps offline imports free of FHIR/configuration.
    Missing counters remain unknown. This is FHIR pagination accounting, not
    accounting for authentication traffic or arbitrary network activity.
    """
    def __init__(self, client_class):
        self.client_class = client_class
        self.records = []
        self.lock = threading.Lock()

    def __enter__(self):
        original = self.client_class._fetch_resources_with_pagination
        self.original = original
        observer = self

        def observed(client, *args, **kwargs):
            try:
                return original(client, *args, **kwargs)
            finally:
                value = getattr(client, 'last_search_stats', None)
                record = {}
                for key in ('physical_requests', 'pages', 'response_bytes', 'returned_records'):
                    count = value.get(key) if isinstance(value, dict) else None
                    record[key] = count if type(count) is int and count >= 0 else None
                record['complete'] = value.get('complete') if isinstance(value, dict) else None
                with observer.lock:
                    observer.records.append(record)
        self.client_class._fetch_resources_with_pagination = observed
        return self

    def __exit__(self, *exception):
        self.client_class._fetch_resources_with_pagination = self.original

    def report(self):
        result = {'observed_physical_queries': len(self.records),
                  'incomplete_queries': sum(row['complete'] is not True for row in self.records),
                  'scope': 'Existing FHIRClient pagination boundaries only; no URLs/patient IDs or auth traffic captured'}
        for key in ('physical_requests', 'pages', 'response_bytes', 'returned_records'):
            values = [row[key] for row in self.records]
            result[key] = sum(values) if all(value is not None for value in values) else None
        result['physical_counts_known'] = all(row[key] is not None for row in self.records
                                               for key in ('physical_requests', 'pages'))
        return result
