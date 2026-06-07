"""Ingest ProteinBase: bulk CSV -> SQLite of measurements, then cache the
kinetic-curve JSONs for the demo targets.

A "measurement" = one (protein, target, curve_url) triple. ProteinBase stores
replicate evaluations as a flat list, so we collapse per (protein, target):
take the binder/strength label and the median kd/kon/koff, and emit one
measurement row per distinct kinetic-curve URL found for that pair.

Run:  python ingest.py            # PD-L1 + il7r (default)
      python ingest.py --all-meta # load every protein's metadata, curves for demo targets only
"""
import argparse
import csv
import json
import sqlite3
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import requests

import config

csv.field_size_limit(10**8)

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,   -- curve uuid (stem of the json)
    protein_id     TEXT,
    name           TEXT,
    author         TEXT,
    design_method  TEXT,
    target         TEXT,
    binder         INTEGER,            -- 1/0/NULL
    binding_strength TEXT,             -- Strong/Medium/Weak/None
    kd REAL, kon REAL, koff REAL,
    fit_model      TEXT,               -- bivalent/standard
    expressed      INTEGER,
    is_control     INTEGER,
    curve_url      TEXT,
    curve_file     TEXT                -- local cached path (relative)
);
CREATE INDEX IF NOT EXISTS idx_target ON measurements(target);
CREATE INDEX IF NOT EXISTS idx_protein ON measurements(protein_id);
"""


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def parse_csv(targets=None):
    """Yield measurement dicts. targets=None ingests every target."""
    targets = set(targets) if targets else None
    with open(config.CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ev = json.loads(row.get("evaluations") or "[]")
            # group by target
            per_target = defaultdict(lambda: {
                "binder": None, "strength": None, "kd": [], "kon": [],
                "koff": [], "fit": None, "expressed": None, "curves": []})
            for e in ev:
                t = e.get("target")
                m = e.get("metric")
                v = e.get("value")
                if m == "expressed" and isinstance(v, bool):
                    for d in per_target.values():
                        d["expressed"] = v
                if not t:
                    continue
                d = per_target[t]
                if m == "binding" and isinstance(v, bool):
                    d["binder"] = v
                elif m == "binding_strength":
                    d["strength"] = v
                elif m == "kd" and isinstance(v, (int, float)):
                    d["kd"].append(v)
                elif m == "kon" and isinstance(v, (int, float)):
                    d["kon"].append(v)
                elif m == "koff" and isinstance(v, (int, float)):
                    d["koff"].append(v)
                elif m == "selected_binding_fit_model":
                    d["fit"] = v
                elif m in ("spr_kinetic_curves", "bli_kinetic_curves"):
                    url = v.get("url") if isinstance(v, dict) else None
                    if url and url.endswith(".json"):
                        d["curves"].append(url)
            name = (row.get("name") or "")
            is_control = 1 if name.lower().startswith("control") or "control" in name.lower()[:8] else 0
            for t, d in per_target.items():
                if (targets is not None and t not in targets) or not d["curves"]:
                    continue
                for url in dict.fromkeys(d["curves"]):   # dedupe, keep order
                    mid = url.rsplit("/", 1)[-1].replace(".json", "")
                    yield {
                        "measurement_id": mid,
                        "protein_id": row.get("id"),
                        "name": name,
                        "author": row.get("author") or "",
                        "design_method": row.get("designMethod") or "",
                        "target": t,
                        "binder": None if d["binder"] is None else int(d["binder"]),
                        "binding_strength": d["strength"],
                        "kd": _median(d["kd"]),
                        "kon": _median(d["kon"]),
                        "koff": _median(d["koff"]),
                        "fit_model": d["fit"],
                        "expressed": None if d["expressed"] is None else int(d["expressed"]),
                        "is_control": is_control,
                        "curve_url": url,
                        "curve_file": f"curves/{mid}.json",
                    }


def download_curve(url):
    mid = url.rsplit("/", 1)[-1]
    dest = config.CURVES / mid
    if dest.exists() and dest.stat().st_size > 0:
        return mid, True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return mid, True
    except Exception as e:
        return f"{mid} ERROR {e}", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None,
                    help="limit to these target slugs (default: ALL targets)")
    ap.add_argument("--download", action="store_true",
                    help="also pre-download all curve JSONs (default: lazy)")
    args = ap.parse_args()

    if not config.CSV.exists():
        url = "https://storage.proteinbase.com/proteinbase_all_data_28_01_2026.csv"
        print(f"Downloading ProteinBase CSV from {url} ...")
        config.CSV.write_bytes(requests.get(url, timeout=300).content)

    rows = list(parse_csv(args.targets))
    print(f"Parsed {len(rows)} measurements across targets={args.targets or 'ALL'}")
    if not rows:
        sys.exit("No measurements found - check target slugs.")

    con = sqlite3.connect(config.DB)
    con.execute("DROP TABLE IF EXISTS measurements")
    con.executescript(SCHEMA)
    cols = list(rows[0].keys())
    con.executemany(
        f"INSERT OR REPLACE INTO measurements ({','.join(cols)}) "
        f"VALUES ({','.join('?' for _ in cols)})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    con.commit()

    if args.download:
        urls = list(dict.fromkeys(r["curve_url"] for r in rows))
        print(f"Downloading {len(urls)} curve files (cached) ...")
        ok = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            for i, (_, good) in enumerate(ex.map(download_curve, urls), 1):
                ok += good
                if i % 200 == 0:
                    print(f"  [{i}/{len(urls)}]")
        print(f"Cached {ok}/{len(urls)} curves.")
    else:
        print("Curves download lazily on first view.")

    n_t = con.execute("SELECT COUNT(DISTINCT target) FROM measurements").fetchone()[0]
    print(f"\n{len(rows)} measurements across {n_t} targets.")
    for t in con.execute("SELECT target, COUNT(*), SUM(binder) FROM measurements "
                         "GROUP BY target ORDER BY COUNT(*) DESC LIMIT 12").fetchall():
        print("  ", t)
    con.close()


if __name__ == "__main__":
    main()
