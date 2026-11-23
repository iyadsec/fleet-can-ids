#!/usr/bin/env python3
"""Verify the snapshot modules against the bytecode compiled from them in June 2026 (run with Python 3.13).

The CPython .pyc files written in June record each module's source size and mtime and hold its compiled code
object. They also embed the absolute path of the June working tree, so they are kept privately and are not
published. bytecode_record.csv holds, for each module, the SHA-256 of the original .pyc, its header fields and
a digest of its code object that excludes the file path (code_digest below).

Default mode compiles each snapshot source, computes the same digest and compares it with the record, then
writes MANIFEST.csv. A module is bytecode-identical when the digests and source sizes match. The digest covers
instructions, constants including docstrings and nested functions, names, line table and exception table;
comments and whitespace that do not change line numbers are not covered.

--bytecode-dir DIR checks original .pyc files against the record: SHA-256, header fields and digest, and
direct code-object equality with the compiled snapshot source. --write-record rebuilds the record from them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import marshal
import struct
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "bytecode_record.csv"
BATCH_START = dt.datetime(2026, 6, 14, 20, 19, 36, tzinfo=dt.timezone.utc)
REQUIRED = set((HERE / "required_by_june_runner.txt").read_text().split())
UTC = dt.timezone.utc
CODE_FIELDS = (
    "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags",
    "co_firstlineno", "co_name", "co_qualname", "co_code", "co_names", "co_varnames", "co_freevars",
    "co_cellvars", "co_linetable", "co_exceptiontable",
)
RECORD_FIELDS = ("path", "pyc_name", "pyc_sha256", "pyc_bytes", "pyc_magic", "bytecode_source_mtime_utc",
                 "bytecode_source_bytes", "code_digest")


def canonical(obj):
    """Hash-seed-independent structure of a code object, without co_filename."""
    if isinstance(obj, types.CodeType):
        return ("code", tuple(canonical(getattr(obj, f)) for f in CODE_FIELDS), canonical(obj.co_consts))
    if isinstance(obj, tuple):
        return ("tuple", tuple(canonical(x) for x in obj))
    if isinstance(obj, frozenset):
        return ("frozenset", tuple(sorted(repr(canonical(x)) for x in obj)))
    return (type(obj).__name__, repr(obj))


def code_digest(code: types.CodeType) -> str:
    return hashlib.sha256(repr(canonical(code)).encode()).hexdigest()


def pyc_name(rel: str) -> str:
    return rel.replace("/", "__").replace(".py", ".cpython-313.pyc")


def read_pyc(path: Path) -> tuple[dict, types.CodeType]:
    b = path.read_bytes()
    if b[:4] != importlib.util.MAGIC_NUMBER:
        raise SystemExit(f"bytecode magic mismatch for {path.name}")
    _, src_mtime, src_size = struct.unpack("<III", b[4:16])
    code = marshal.loads(b[16:])
    fields = dict(pyc_name=path.name, pyc_sha256=hashlib.sha256(b).hexdigest(), pyc_bytes=len(b),
                  pyc_magic=b[:4].hex(), bytecode_source_mtime_utc=dt.datetime.fromtimestamp(src_mtime, UTC).isoformat(),
                  bytecode_source_bytes=src_size, code_digest=code_digest(code))
    return fields, code


def check_bytecode(bytecode_dir: Path, sources: list[Path], record: dict, write: bool) -> None:
    rows = []
    for src in sources:
        rel = str(src.relative_to(HERE))
        fields, code = read_pyc(bytecode_dir / pyc_name(rel))
        data = src.read_bytes()
        equal = compile(data, code.co_filename, "exec", dont_inherit=True) == code
        same_digest = code_digest(compile(data, rel, "exec", dont_inherit=True)) == fields["code_digest"]
        if equal != same_digest:
            raise SystemExit(f"code-object equality and digest disagree for {rel}")
        if not write and {k: str(v) for k, v in fields.items()} != {k: record[rel][k] for k in fields}:
            raise SystemExit(f"{fields['pyc_name']} does not match bytecode_record.csv")
        rows.append(dict(path=rel, **fields))
    if write:
        with RECORD.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RECORD_FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"original bytecode: {len(rows)} files match {RECORD.name}; digest agrees with code-object equality")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytecode-dir", type=Path, help="directory holding the original June .pyc files")
    ap.add_argument("--write-record", action="store_true", help="rebuild bytecode_record.csv from --bytecode-dir")
    args = ap.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python 3.13 is required to compile code comparable with the June bytecode")
    if args.write_record and not args.bytecode_dir:
        raise SystemExit("--write-record needs --bytecode-dir")
    sources = sorted((HERE / "src").rglob("*.py"))
    record = {}
    if RECORD.exists():
        with RECORD.open(newline="") as fh:
            record = {r["path"]: r for r in csv.DictReader(fh)}
    if args.bytecode_dir:
        check_bytecode(args.bytecode_dir, sources, record, args.write_record)
        with RECORD.open(newline="") as fh:
            record = {r["path"]: r for r in csv.DictReader(fh)}
    rows = []
    for src in sources:
        rel = str(src.relative_to(HERE))
        data = src.read_bytes()
        rec = record[rel]
        if rec["pyc_magic"] != importlib.util.MAGIC_NUMBER.hex():
            raise SystemExit(f"bytecode magic mismatch for {rel}")
        digest = code_digest(compile(data, rel, "exec", dont_inherit=True))
        src_size = int(rec["bytecode_source_bytes"])
        recorded = dt.datetime.fromisoformat(rec["bytecode_source_mtime_utc"])
        row = dict(path=rel, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                   required_by_june_runner="yes" if rel in REQUIRED else "no",
                   bytecode_source_mtime_utc=recorded.isoformat(), bytecode_source_bytes=src_size,
                   bytecode_identical=digest == rec["code_digest"] and src_size == len(data))
        if not row["bytecode_identical"]:
            status = "UNVERIFIED: source differs from June bytecode"
        elif recorded < BATCH_START:
            status = "verified: identical to bytecode of source dated before the June batch"
        else:
            status = "UNVERIFIED: matches only bytecode recompiled after the June batch"
        row["status"] = status
        rows.append(row)
    with (HERE / "MANIFEST.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['status'][:10]:10}  req={r['required_by_june_runner']:3}  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
