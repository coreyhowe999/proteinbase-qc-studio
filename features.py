"""Per-measurement QC feature extraction from cached kinetic-curve JSONs.

A curve file is a dict of {series_id: series}, one series per analyte
concentration. Each series has raw {t, y}, a concentration, control flag, and
(usually) fits {standard, bivalent} -> {association, dissociation} -> {t, y}.

featurize() returns a flat dict of numeric/boolean features that map to the
QC vocabulary Adaptyv already uses (staircase, carryover, weak signal, drift,
saturation, low loading, unexpected order, fit quality). These features are the
shared language the natural-language check builder composes checks from.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import config


def load_curve(measurement_id: str) -> dict:
    """Load a kinetic-curve JSON. Downloads from ProteinBase's public CDN and
    caches locally on first miss (so it works on hosts that ship no curve cache,
    e.g. Streamlit Cloud)."""
    p = config.CURVES / f"{measurement_id}.json"
    if not p.exists() or p.stat().st_size == 0:
        import requests
        r = requests.get(config.CURVE_BASE + measurement_id + ".json", timeout=30)
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
    return json.loads(p.read_text())


def _arr(d, k):
    return np.asarray(d.get(k, []), dtype=float)


def _r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) < 3:
        return None
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def _series_features(series: dict) -> dict | None:
    """Features for one concentration trace."""
    raw = series.get("raw") or {}
    t, y = _arr(raw, "t"), _arr(raw, "y")
    if len(t) < 10 or len(y) != len(t):
        return None
    n = len(t)

    # Two shapes exist in the data:
    #   series["fit"]  = {association, dissociation}                 (single model)
    #   series["fits"] = {standard|bivalent: {association, dissociation}}
    fit, model = None, None
    if isinstance(series.get("fit"), dict) and series["fit"].get("association"):
        fit, model = series["fit"], series.get("fit_model") or "fitted"
    elif isinstance(series.get("fits"), dict) and series["fits"]:
        fits = series["fits"]
        model = "standard" if "standard" in fits else next(iter(fits))
        fit = fits.get(model)

    # association end time = end of the association fit window if available
    assoc_end_t = None
    if fit and fit.get("association", {}).get("t"):
        assoc_end_t = float(np.max(_arr(fit["association"], "t")))
    if assoc_end_t is None:
        assoc_end_t = float(t[int(0.55 * n)])  # heuristic split

    baseline = float(np.mean(y[: max(3, n // 33)]))
    noise = float(np.std(y[: max(3, n // 33)]))

    assoc_mask = t <= assoc_end_t
    diss_mask = t > assoc_end_t
    # plateau / equilibrium response = mean of last 10% of the association phase
    if assoc_mask.sum() >= 5:
        a_y = y[assoc_mask]
        plateau = float(np.mean(a_y[-max(3, len(a_y) // 10):]))
    else:
        plateau = float(np.max(y))
    rmax = float(np.max(y))

    # end-of-dissociation level (carryover / return-to-baseline)
    if diss_mask.sum() >= 3:
        end_level = float(np.mean(y[diss_mask][-max(3, int(diss_mask.sum() * 0.03)):]))
    else:
        end_level = float(y[-1])

    # baseline drift: slope of a line fit over the first 10% of the trace (nm/s)
    k = max(5, n // 10)
    drift_slope = float(np.polyfit(t[:k] - t[0], y[:k], 1)[0])

    # dissociation retention: how much signal is still bound at the end of the
    # dissociation phase relative to the end of association. ~1.0 = the complex
    # barely comes off (slow off-rate). Normal binders drop well below 1.
    span = plateau - baseline
    dissoc_retention = float((end_level - baseline) / span) if abs(span) > 1e-6 else None

    # association completion: fraction of the plateau reached by the MIDPOINT of
    # the association phase. Low (<~0.7) = a slow, creeping association that does
    # not saturate within the contact time (slow on-rate / aggregation).
    assoc_completion = None
    if assoc_mask.sum() >= 6:
        ta, ya = t[assoc_mask], y[assoc_mask]
        tmid = ta[0] + (ta[-1] - ta[0]) * 0.5
        rmid = float(np.interp(tmid, ta, ya))
        if abs(plateau - baseline) > 1e-6:
            assoc_completion = float((rmid - baseline) / (plateau - baseline))

    # fit quality on this trace: compare raw vs fit over the fit's own t-grid
    r2 = None
    if fit:
        fy, ft = [], []
        for phase in ("association", "dissociation"):
            ph = fit.get(phase) or {}
            pt, py = _arr(ph, "t"), _arr(ph, "y")
            if len(pt) and len(pt) == len(py):
                ft.append(pt); fy.append(py)
        if ft:
            ft = np.concatenate(ft); fy = np.concatenate(fy)
            yi = np.interp(ft, t, y)
            r2 = _r2(yi, fy)

    return dict(
        concentration=float(series.get("concentration") or 0.0),
        is_control=bool(series.get("control")),
        baseline=baseline, noise=noise, plateau=plateau, rmax=rmax,
        end_level=end_level, drift_slope=drift_slope, assoc_end_t=assoc_end_t,
        fit_model=model, r2=r2, n_points=n,
        carryover_frac=float((end_level - baseline) / rmax) if rmax > 1e-9 else None,
        dissoc_retention=dissoc_retention, assoc_completion=assoc_completion,
    )


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else None


# overall-signal scale used to call "weak signal" / "low loading" (nm).
WEAK_SIGNAL_NM = 0.5
SNR_WEAK = 5.0


def featurize(curve: dict) -> dict:
    """Measurement-level QC features aggregated across the concentration series."""
    series = [s for s in (_series_features(v) for v in curve.values()) if s]
    series.sort(key=lambda s: s["concentration"])
    if not series:
        return {"n_concentrations": 0, "valid": False}

    concs = [s["concentration"] for s in series]
    plateaus = [s["plateau"] for s in series]
    rmax_all = max(s["rmax"] for s in series)
    noise = float(np.median([s["noise"] for s in series]))
    r2s = [s["r2"] for s in series if s["r2"] is not None]

    # staircase / unexpected order: response should grow with concentration
    rho = _spearman(concs, plateaus) if len(series) > 1 else None
    inversions = sum(
        1 for i in range(1, len(plateaus))
        if plateaus[i] < plateaus[i - 1] - 0.05 * max(rmax_all, 1e-9)
    )
    order_respected = (rho is None) or (rho >= 0.9 and inversions == 0)

    # saturation: top concentration adds little response vs the one below it
    sat_ratio = None
    if len(plateaus) >= 2 and plateaus[-2] > 1e-6:
        sat_ratio = float((plateaus[-1] - plateaus[-2]) / abs(plateaus[-2]))

    # use the strongest-signal trace for shape metrics (best SNR)
    strong = max(series, key=lambda s: s["rmax"])
    carryover = strong["carryover_frac"]
    dissoc_retention = strong.get("dissoc_retention")
    assoc_completion = strong.get("assoc_completion")

    # drift: largest baseline slope magnitude across the series
    drift = max(abs(s["drift_slope"]) for s in series)

    snr = float(rmax_all / noise) if noise > 1e-9 else float("inf")

    return {
        "valid": True,
        "n_concentrations": len(series),
        "concentrations": concs,
        "plateau_by_conc": plateaus,
        "max_response": rmax_all,
        "noise": noise,
        "snr": snr,
        "min_r2": min(r2s) if r2s else None,
        "mean_r2": float(np.mean(r2s)) if r2s else None,
        "spearman_conc_response": rho,
        "n_order_inversions": inversions,
        "order_respected": order_respected,
        "saturation_top_step_frac": sat_ratio,
        "carryover_frac": carryover,
        "dissoc_retention": dissoc_retention,
        "assoc_completion": assoc_completion,
        "max_baseline_drift": drift,
        "weak_signal": bool(rmax_all < WEAK_SIGNAL_NM or snr < SNR_WEAK),
        "fit_model": series[-1]["fit_model"],
        "series": series,  # kept for plotting / code-mode checks
    }


FEATURE_GLOSSARY = """\
Available per-measurement features (computed across the concentration series):
- n_concentrations (int): number of analyte concentrations in the series
- concentrations (list[float], nM): sorted low->high
- plateau_by_conc (list[float], nm): equilibrium response at each concentration (same order)
- max_response (float, nm): largest binding signal observed
- noise (float, nm): median baseline noise
- snr (float): max_response / noise
- min_r2 / mean_r2 (float): fit quality of the 1:1 model vs raw trace (1.0 = perfect)
- spearman_conc_response (float, -1..1): rank correlation of concentration vs response.
    ~1.0 = textbook staircase; low/negative = response order does NOT track concentration order
- n_order_inversions (int): count of concentration steps where response went DOWN
- order_respected (bool): True if response increases monotonically with concentration
- saturation_top_step_frac (float): fractional response gain from 2nd-highest to highest
    concentration. Near 0 = saturated (more analyte stops adding signal)
- carryover_frac (float): residual signal at end of dissociation / max_response.
    High = complex does not fully dissociate (carryover)
- dissoc_retention (float): signal still bound at the END of dissociation divided by
    the end-of-association response. ~1.0 = the complex barely comes off (slow off-rate);
    normal binders fall well below 1. (For slow-on/slow-off "aggregation" behaviour.)
- assoc_completion (float): fraction of the plateau reached by the MIDPOINT of the
    association phase. Low (<~0.7) = slow, creeping association that doesn't saturate
    in the contact time (slow on-rate)
- kon (float, 1/Ms), koff (float, 1/s), kd (float, M): fitted kinetics where available
    (may be null). Slow on-rate = low kon (<~1e4); slow off-rate = low koff (<~1e-4)
- max_baseline_drift (float, nm/s): steepest baseline slope (drift)
- weak_signal (bool): max_response below ~0.5 nm or SNR below ~5
"""
