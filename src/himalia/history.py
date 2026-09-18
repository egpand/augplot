"""Persistent, versioned source history. Never deserialize executable Python objects."""

import datetime as dt
import hashlib
import json
import os
import struct
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ConfigurationError, DataError

HISTORY_VERSION = 1


def fingerprint(data):
    """Hash all values, ordering, and plotting-relevant schema, not a sampled profile."""
    digest = hashlib.sha256()

    def token(value):
        payload = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def dtype(value):
        token(str(value))
        if isinstance(value, pd.CategoricalDtype):
            visit(value.categories)
            visit(value.ordered)

    def visit(value):
        token(f"{type(value).__module__}.{type(value).__qualname__}")
        if value is None or value is pd.NA or value is pd.NaT:
            return
        if isinstance(value, pd.DataFrame):
            visit(value.index)
            visit(value.columns)
            for i in range(len(value.columns)):
                visit(value.iloc[:, i])
        elif isinstance(value, (pd.Series, pd.Index)):
            if isinstance(value, pd.Series):
                visit(value.index)
                visit(value.name)
            else:
                visit(list(value.names))
                token(str(getattr(value, "freqstr", None)))
                if isinstance(value, pd.MultiIndex):
                    for level in value.levels:
                        visit(level)
            dtype(value.dtype)
            token(len(value))
            for item in value:
                visit(item)
        elif isinstance(value, np.ndarray):
            token(value.dtype.str)
            token(repr(value.dtype.descr))
            visit(value.shape)
            if value.dtype.hasobject:
                for item in value.flat:
                    visit(item)
            else:
                token(value.tobytes(order="C"))
        elif isinstance(value, np.generic):
            token(value.dtype.str)
            token(repr(value.dtype.descr))
            if value.dtype.hasobject:
                visit(value.tolist())
            else:
                token(value.tobytes())
        elif isinstance(value, dict):
            token(len(value))
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list)):
            token(len(value))
            for item in value:
                visit(item)
        elif isinstance(value, (str, bool, int)):
            token(value)
        elif isinstance(value, float):
            token(struct.pack("!d", value))
        elif isinstance(value, complex):
            token(struct.pack("!dd", value.real, value.imag))
        elif isinstance(value, pd.Period):
            token(value.ordinal)
            token(value.freqstr)
        elif isinstance(value, pd.Interval):
            visit(value.left)
            visit(value.right)
            token(value.closed)
        elif isinstance(value, dt.date):
            token(value.isoformat())
            token(getattr(value, "fold", 0))
            token(str(getattr(value, "tzinfo", None)))
        elif isinstance(value, dt.timedelta):
            token(str(value))
        else:
            raise DataError("Cannot fingerprint this input; convert it to ordinary data first.")

    visit(data)
    return digest.hexdigest()


def request_key(settings):
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class History:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()

    def load(self, root, key):
        entry = self.directory / root / f"{key}.json"
        if not entry.exists():
            return None
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
            if (
                record["history_version"] != HISTORY_VERSION
                or record["request_key"] != key
                or record["root"] != root
                or not isinstance(record["explanation"], str)
                or not isinstance(record["depth"], int)
                or record["depth"] < 0
                or not isinstance(record["revision"], str)
            ):
                raise ValueError("Invalid history record")
            filename = record["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Invalid source path")
            source = entry.parent / filename
            code = source.read_text(encoding="utf-8")
            if hashlib.sha256(code.encode("utf-8")).hexdigest() != record["code_hash"]:
                raise ValueError("Source checksum mismatch")
            return record, code, source
        except (OSError, ValueError, KeyError, TypeError):
            raise ConfigurationError(
                "Saved visualization history is unreadable or changed. Restore it or "
                "pass regenerate=True to explicitly generate a new version."
            ) from None

    def save(self, root, key, *, code, explanation, depth, parent, data_fingerprint):
        directory = self.directory / root
        revision = uuid.uuid4().hex
        label = "initial" if depth == 0 else f"refined_{depth}"
        filename = f"{label}_{revision}.py"
        record = {
            "history_version": HISTORY_VERSION,
            "root": root,
            "request_key": key,
            "revision": revision,
            "parent": parent,
            "data_fingerprint": data_fingerprint,
            "depth": depth,
            "filename": filename,
            "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "explanation": explanation,
        }
        temporary = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / filename
            # Immutable source and metadata preserve earlier explicitly regenerated versions.
            source.write_text(code, encoding="utf-8")
            serialized = json.dumps(record, indent=2) + "\n"
            (directory / f"{label}_{revision}.json").write_text(serialized, encoding="utf-8")
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            # Publish the lookup only after the complete source has been written.
            os.replace(temporary, directory / f"{key}.json")
        except OSError:
            raise ConfigurationError(
                "Could not save visualization history. Choose a writable cache_dir "
                "or use cache_dir=None to disable persistence."
            ) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return record, code, source
