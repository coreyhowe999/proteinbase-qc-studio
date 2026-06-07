"""Download + featurize ALL ProteinBase curves into the feature cache so QC
checks can filter across the entire dataset. Idempotent; resumes from cache."""
import sqlite3, time, config, dataio
con = sqlite3.connect(config.DB)
rows = con.execute("SELECT measurement_id, curve_url FROM measurements").fetchall()
url_map = {m: u for m, u in rows if u}
print(f"precomputing features for {len(url_map)} measurements...")
t = time.time()
feats = dataio.ensure_features(url_map, max_workers=16)
valid = sum(1 for f in feats.values() if f.get("valid"))
print(f"done: {valid}/{len(feats)} valid in {round(time.time()-t)}s")
