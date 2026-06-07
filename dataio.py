"""Lazy curve + feature loading with an on-disk cache.

ProteinBase has ~6500 curves (~1 GB). We ingest only metadata eagerly; curve
JSONs are downloaded the first time they're needed, and their QC features are
computed once and cached in SQLite. This lets the viewer cover ALL experiments
without a giant up-front download.
"""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import requests

import config
import features as featmod

_SESS = requests.Session()


def ensure_schema():
    con = sqlite3.connect(config.DB, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS feature_cache "
                "(measurement_id TEXT PRIMARY KEY, json TEXT)")
    con.commit()
    con.close()


def curve_path(mid):
    return config.CURVES / f"{mid}.json"


def get_curve(mid: str, url: str | None = None) -> dict:
    p = curve_path(mid)
    if not p.exists() or p.stat().st_size == 0:
        if not url:
            raise FileNotFoundError(f"curve {mid} not cached and no url given")
        r = _SESS.get(url, timeout=30)
        r.raise_for_status()
        p.write_bytes(r.content)
    return json.loads(p.read_text())


def _compute(mid, url):
    try:
        return mid, featmod.featurize(get_curve(mid, url))
    except Exception:
        return mid, {"valid": False}


def ensure_features(url_map: dict[str, str], max_workers: int = 12) -> dict[str, dict]:
    """Return {mid: features} for every mid in url_map, using/filling the cache.

    url_map maps measurement_id -> curve_url. Cached entries are read in bulk;
    missing ones are downloaded + featurized in parallel, then cached.
    """
    ensure_schema()
    ids = list(url_map)
    out: dict[str, dict] = {}
    con = sqlite3.connect(config.DB, timeout=30)
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        rows = con.execute(
            "SELECT measurement_id, json FROM feature_cache WHERE measurement_id IN "
            f"({','.join('?' * len(chunk))})", chunk).fetchall()
        for mid, js in rows:
            out[mid] = json.loads(js)
    con.close()

    missing = [m for m in ids if m not in out]
    if missing:
        computed = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for mid, f in ex.map(lambda m: _compute(m, url_map[m]), missing):
                computed[mid] = f
                out[mid] = f
        con = sqlite3.connect(config.DB, timeout=30)
        con.executemany("INSERT OR REPLACE INTO feature_cache VALUES (?, ?)",
                        [(m, json.dumps(f)) for m, f in computed.items()])
        con.commit()
        con.close()
    return out
