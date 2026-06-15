"""SHA-256 checksums for authoritative data inputs.

Each run manifest must record `feba_db_sha256`, `workbook_v4_sha256`,
`embedding_manifest_id`, and `canonical_manifest_id` per the data-contract
hard gate (ARCHITECTURE.md §3). Computing SHA-256 on multi-GB files is slow
(~30s for feba.db); we cache results in artifacts/checksums_cache.json
keyed by (path, mtime, size).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable


CACHE_PATH = Path("artifacts/checksums_cache.json")
CHUNK_SIZE = 1 << 20   # 1 MiB


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def file_sha256(path: Path | str, *, use_cache: bool = True) -> str:
    """Return SHA-256 of a file. Caches by (path, mtime, size).

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checksum target missing: {path}")
    stat = path.stat()
    cache_key = str(path.resolve())
    cache = _load_cache() if use_cache else {}
    entry = cache.get(cache_key)
    if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
        return entry["sha256"]
    digest = _file_sha256(path)
    if use_cache:
        cache[cache_key] = {
            "sha256": digest,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
        _save_cache(cache)
    return digest


def manifest_sha256(paths: Iterable[Path | str], *, use_cache: bool = True) -> tuple[str, dict]:
    """Compute a manifest SHA-256 over a set of files.

    Returns (manifest_sha256, per_file_dict) where per_file_dict maps
    relative-or-absolute path → file sha256.

    The manifest digest is the SHA-256 of the canonical JSON
    (sorted keys, no whitespace) of per_file_dict.
    """
    per_file: dict[str, str] = {}
    for p in sorted(map(str, paths)):
        per_file[p] = file_sha256(p, use_cache=use_cache)
    canonical = json.dumps(per_file, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), per_file


def short(digest: str, n: int = 7) -> str:
    """Short form of a digest, for run-id construction."""
    return digest[:n]
