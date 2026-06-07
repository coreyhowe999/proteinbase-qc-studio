"""Build experiment (= ProteinBase collection) -> protein membership.

The bulk CSV has no experiment field, but each collection page embeds a
collectionId, and /api/proteins/download?collectionId=... returns that
collection's proteins. We fetch each collection, scoped-download its proteins,
and record protein ids + targets. Output: data/experiments.json
"""
import csv
import io
import json
import re

import requests

import config

csv.field_size_limit(10**8)

# slug -> display name (from proteinbase.com/collections)
COLLECTIONS = {
    "adaptyv-x-muni-hackathon-ai-agents-vs-humans": "Adaptyv x muni hackathon (AI vs Humans)",
    "pd-l1-foldcraft": "PD-L1 FoldCraft",
    "boolean-biotech-vhh-competition-2025": "Boolean Biotech VHH Competition 2025",
    "evolved-hackathon": "Evolved Hackathon",
    "gem-x-adaptyv-rbx1-binder-design-competition-results": "GEM x Adaptyv RBX1 Competition",
    "nipah-binder-competition-results": "Nipah Competition Results",
    "nipah-binder-competition-all-submissions": "Nipah Competition (All Submissions)",
    "mog-dfm-spotlight": "MOG-DFM Spotlight",
    "boltzgen-release": "BoltzGen Release",
    "cradle-egfr-competition": "Cradle EGFR Competition Follow Up",
    "adaptyv-egfr-competition-round-1": "Adaptyv EGFR Competition Round 1",
    "mosaic-development": "Mosaic Development",
    "protrl-validation": "ProtRL Validation",
    "mosaic-multispecifics": "Mosaic Multispecifics",
    "bindcraft1-revalidation": "BindCraft1 publication re-validation",
    "evolved-2024-bio-x-ml-team-silica-egfr-nanobodies": "Evolved 2024 Silica EGFR Nanobodies",
    "egfr-round1-second-submission": "EGFR Round 1 Second Submission",
    "dsm-round-1": "DSM Synteract Round 1",
    "adaptyv-egfr-competition-round-2": "Adaptyv EGFR Competition Round 2",
    "rfdiffusion-re-validation": "RFdiffusion re-validation",
    "pro-1-validation": "Pro-1 validation",
}

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (QC-Studio ingest)"


_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def collection_id(slug):
    html = S.get(f"https://proteinbase.com/collections/{slug}", timeout=40).text
    # the download link carries the right id: download?collectionId=<uuid>&slug=
    m = re.search(rf"download\?collectionId=({_UUID})", html)
    if not m:
        m = re.search(rf"collectionId[\\\":=\s]{{1,6}}({_UUID})", html, re.I)
    return m.group(1) if m else None


def scoped_proteins(cid, slug):
    url = f"https://proteinbase.com/api/proteins/download?collectionId={cid}&slug={slug}"
    txt = S.get(url, timeout=120).text.lstrip("﻿")
    pids, targets = [], set()
    for row in csv.DictReader(io.StringIO(txt)):
        pids.append(row["id"])
        for e in json.loads(row.get("evaluations") or "[]"):
            if e.get("target"):
                targets.add(e["target"])
    return pids, sorted(targets)


def main():
    out = {}
    for slug, name in COLLECTIONS.items():
        try:
            cid = collection_id(slug)
            if not cid:
                print(f"  ! no collectionId for {slug}")
                continue
            pids, targets = scoped_proteins(cid, slug)
            out[name] = {"slug": slug, "collectionId": cid,
                         "targets": targets, "protein_ids": pids}
            print(f"  {name}: {len(pids)} proteins, targets={targets}")
        except Exception as e:
            print(f"  ! {slug}: {e}")
    (config.DATA / "experiments.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\nWrote {len(out)} experiments to data/experiments.json")


if __name__ == "__main__":
    main()
