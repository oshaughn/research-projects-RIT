"""Collect the explicit core manifest once and report exact per-file item counts."""
import argparse
import os
from pathlib import Path
import sys


def count_nodeids(files, nodeids, rootpath, cwd=None):
    """Count actual pytest items, never human-readable collection output."""
    cwd = Path.cwd() if cwd is None else Path(cwd)
    rootpath = Path(rootpath).resolve(strict=True)
    counts = {}
    owners = {}
    for name in files:
        if not isinstance(name, str) or not name or any(c in name for c in '\t\n\r\0'):
            raise ValueError('invalid manifest file')
        path = (cwd / name).resolve(strict=True)
        if path in owners:
            raise ValueError('duplicate manifest file')
        owners[path] = name
        counts[name] = 0
    if not counts:
        raise ValueError('empty manifest')
    seen = set()
    for nodeid in nodeids:
        if (not isinstance(nodeid, str) or not nodeid or '::' not in nodeid
                or any(c in nodeid for c in '\n\r\0')):
            raise ValueError('malformed collected nodeid')
        filename, item = nodeid.split('::', 1)
        path = Path(filename)
        if not filename or not item or '\\' in filename or path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid nodeid file or item')
        if nodeid in seen:
            raise ValueError('duplicate collected nodeid')
        seen.add(nodeid)
        owner = owners.get((rootpath / path).resolve(strict=True))
        if owner is None:
            raise ValueError('collected item outside explicit manifest: ' + nodeid)
        counts[owner] += 1
    if not seen or sum(counts.values()) != len(seen):
        raise ValueError('empty or inconsistent collection')
    return counts


class CollectedItems:
    def __init__(self):
        self.nodeids = None
        self.rootpath = None

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]
        self.rootpath = session.config.rootpath


def collect(files, output):
    # Match python -m pytest: script invocation otherwise puts .travis first.
    sys.path.insert(0, str(Path.cwd()))
    import pytest
    observer = CollectedItems()
    code = int(pytest.main(['--collect-only', '-q', '-p', 'no:cacheprovider', *files],
                          plugins=[observer]))
    if code != 0:
        raise ValueError('joint pytest collection failed with exit ' + str(code))
    if observer.nodeids is None or observer.rootpath is None:
        raise ValueError('collection observer did not finish')
    counts = count_nodeids(files, observer.nodeids, observer.rootpath)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        for name in files:
            stream.write(name + '\t' + str(counts[name]) + '\n')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    try:
        collect(args.files, args.output)
    except (ValueError, OSError) as exc:
        print('core collection: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
