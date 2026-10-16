#!/usr/bin/env python3
"""Bind selected smoke executables to exact builds without copying private inputs."""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def tuple_now():
    return {
        'cg_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'contextdb_sha': re.search(r'contextdb[?]rev=([0-9a-f]{40})', pathlib.Path('Cargo.lock').read_text())[1],
    }


def verify(receipt, binary, expected):
    data = json.loads(pathlib.Path(receipt).read_text())
    if any(data.get(k) != v for k, v in expected.items()) or data.get('binary_sha256') != sha(binary):
        raise ValueError('installed binary build receipt mismatch')
    if data.get('build_exit_code') != 0:
        raise ValueError('installed binary build did not succeed')
    return data


if __name__ == '__main__':
    try:
        mode, binary, receipt = sys.argv[1:4]
        if mode == 'build':
            data = dict(tuple_now(), binary_sha256=sha(binary), build_exit_code=0, command='cargo build --locked --release -p context-graph-cli --bin cg')
            pathlib.Path(receipt).write_text(json.dumps(data, sort_keys=True) + '\n')
        elif mode == 'verify':
            verify(receipt, binary, tuple_now())
        elif mode == 'dependency':
            verify(receipt, binary, {'source_sha': tuple_now()['contextdb_sha']})
        elif mode == 'result':
            data = dict(tuple_now(), binary_sha256=sha(binary), smoke=sys.argv[4], exit_code=int(sys.argv[5]))
            if sys.argv[4] == 'cg-capture':
                data['contextdb_binary_sha256'] = sha(os.environ['CONTEXTDB'])
            pathlib.Path(receipt).write_text(json.dumps(data, sort_keys=True) + '\n')
        else:
            raise ValueError('unknown receipt mode')
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('PENDING: ' + str(error)) from error
