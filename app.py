"""ProteinBase QC Studio - binding-assay data review & QC.

Layout (Browse):
  sidebar     : Experiment (collection) multiselect -> Target select (cascades),
                binder filter, search, collapse-replicates, QC checks
  main / left : measurements table  +  large sensorgram viewer + QC-feature row
  main / lower: same-target gallery (6x4) of clickable sensorgrams to scan patterns
  right rail  : persistent natural-language Check Builder (always visible),
                showing exactly what Claude sees - the curve image + its data
"""
from __future__ import annotations

import base64
import json
import sqlite3

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from st_clickable_images import clickable_images

# keep empty/NaN numeric cells (e.g. KD with no value) at the BOTTOM in either
# sort direction, and render them blank
_NULLS_LAST = JsCode("""
function(a, b, nodeA, nodeB, isInverted) {
    var an = (a === null || a === undefined || a === '' || (typeof a === 'number' && isNaN(a)));
    var bn = (b === null || b === undefined || b === '' || (typeof b === 'number' && isNaN(b)));
    if (an && bn) return 0;
    if (an) return isInverted ? -1 : 1;
    if (bn) return isInverted ? 1 : -1;
    return a - b;
}
""")
_BLANK_NULL = JsCode("function(p){var v=p.value; return (v===null||v===undefined||"
                     "(typeof v==='number'&&isNaN(v)))?'':v;}")

import config
import dataio
import engine
import features as featmod
import viz

st.set_page_config(page_title="ProteinBase QC Studio", layout="wide", page_icon="🧬")

MAX_FEAT = 250        # cap features computed per view (lazy + cached)
GRID_COLS, GRID_ROWS = 6, 4

# --------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_measurements() -> pd.DataFrame:
    con = sqlite3.connect(config.DB)
    df = pd.read_sql("SELECT * FROM measurements", con)
    con.close()
    return df


@st.cache_data(show_spinner=False)
def load_experiments() -> dict:
    p = config.DATA / "experiments.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def png_thumb(mid: str, w: int = 300, h: int = 200, dark: bool = False) -> bytes:
    return viz.sensorgram_png(mid, width=w, height=h, dark=dark)


@st.cache_data(show_spinner=False)
def thumb_datauri(mid: str) -> str:
    return "data:image/png;base64," + base64.b64encode(png_thumb(mid, 300, 200)).decode()


RAIL_BG = "#e6eaf0"   # same grey family as the sidebar


def theme_css() -> str:
    # subtle tint for the QC Builder chatbox panel below the table
    return (f"<style>div[data-testid='stColumn']:has(.builderbox),"
            f"div:has(>.builderbox){{}} "
            f".builder-panel{{background:{RAIL_BG};border-radius:10px;padding:12px 16px;}}</style>")


@st.cache_data(show_spinner="Loading QC features for all measurements…")
def load_all_features(n_rows: int) -> dict:
    """All precomputed features (instant from cache), with fitted kinetics merged
    in so checks can also reference kon/koff/kd. n_rows just keys the cache."""
    con = sqlite3.connect(config.DB)
    kin = {m: (kon, koff, kd) for m, kon, koff, kd in
           con.execute("SELECT measurement_id, kon, koff, kd FROM measurements")}
    con.close()
    feats = dataio.ensure_features(URL)
    for m, f in feats.items():
        if f.get("valid") and m in kin:
            f["kon"], f["koff"], f["kd"] = kin[m]
    return feats


def fmt_kd(kd):
    if kd is None or pd.isna(kd):
        return "-"
    nm = kd * 1e9
    if nm < 1:
        return f"{nm*1000:.0f} pM"
    if nm < 1000:
        return f"{nm:.1f} nM"
    return f"{nm/1000:.2f} µM"


def short_name(name, n=22):
    name = str(name or "")
    if len(name) > 28 and name[:28].isalpha() and name[:28].isupper():
        return name[:10] + "…(seq)"
    return name if len(name) <= n else name[:n] + "…"


df = load_measurements()
EXP = load_experiments()
URL = dict(zip(df["measurement_id"], df["curve_url"]))
PROT_NAME = dict(zip(df["measurement_id"], df["name"]))
MTARGET = dict(zip(df["measurement_id"], df["target"]))
rep_counts = df.groupby(["protein_id", "target"]).size().to_dict()

# protein -> experiment(s) (ProteinBase collection membership)
_prot_exp = {}
for _ename, _info in EXP.items():
    for _pid in _info.get("protein_ids", []):
        _prot_exp.setdefault(_pid, []).append(_ename)
MEXP = {mid: ", ".join(_prot_exp.get(pid, [])) or "—"
        for mid, pid in zip(df["measurement_id"], df["protein_id"])}

ss = st.session_state
ss.setdefault("selected", None)
ss.setdefault("draft", None)
ss.setdefault("rail_open", True)
ss.setdefault("flag_override", {})   # measurement_id -> bool manual flag override

DARK = False
PLOTLY_T = "plotly_white"
AGGRID_THEME = "streamlit"
st.markdown(theme_css(), unsafe_allow_html=True)

# apply pending filter state from a "Save & apply as filter" action - must run
# BEFORE the sidebar widgets are instantiated (can't set a widget key afterwards)
for src, dst in [("_pending_target", "target_sel"), ("_pending_exp", "exp_select")]:
    if src in ss:
        ss[dst] = ss.pop(src)
if "_pending_check" in ss:
    ss[f"chk_{ss.pop('_pending_check')}"] = True

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 QC Studio")
st.sidebar.caption("ProteinBase binding-assay review")

exp_names = sorted(EXP.keys())
sel_exps = st.sidebar.multiselect("Experiments", exp_names, key="exp_select",
                                  help="ProteinBase collections. Leave empty for all.")

# proteins in the selected experiments (or all)
if sel_exps:
    exp_prots = set().union(*(set(EXP[e]["protein_ids"]) for e in sel_exps))
    exp_targets = sorted(set().union(*(set(EXP[e]["targets"]) for e in sel_exps)))
else:
    exp_prots = None
    exp_targets = sorted(df["target"].dropna().unique())

target_opts = ["All targets"] + exp_targets
# persist the choice across reruns; default to pd-l1, reset only if no longer valid
ss.setdefault("target_sel", "pd-l1" if "pd-l1" in target_opts else target_opts[0])
if ss.target_sel not in target_opts:
    ss.target_sel = target_opts[0]
sel_target = st.sidebar.selectbox("Target", target_opts, key="target_sel")
all_targets = sel_target == "All targets"
sel_kind = st.sidebar.radio("Show", ["All", "Binders", "Non-binders"], horizontal=True)
search = st.sidebar.text_input("Search name / id", "")
collapse_reps = st.sidebar.checkbox("Collapse replicates (one row per protein)")

# --------------------------------------------------------------- filtering
fdf = df if all_targets else df[df["target"] == sel_target]
if exp_prots is not None:
    fdf = fdf[fdf["protein_id"].isin(exp_prots)]
if sel_kind == "Binders":
    fdf = fdf[fdf["binder"] == 1]
elif sel_kind == "Non-binders":
    fdf = fdf[fdf["binder"] == 0]
if search.strip():
    s = search.strip().lower()
    fdf = fdf[fdf.apply(lambda r: s in str(r["name"]).lower() or s in str(r["measurement_id"]).lower(), axis=1)]
if collapse_reps:
    fdf = fdf.drop_duplicates(subset=["protein_id", "target"], keep="first")
fdf = fdf.reset_index(drop=True)

capped = False
work = fdf

# gallery pool: all curves for this target (across experiments); or the filtered
# set when "All targets" is chosen
if all_targets:
    gallery_pool = fdf["measurement_id"].tolist()
else:
    gallery_pool = df[df["target"] == sel_target]["measurement_id"].tolist()

# reset gallery paging when the target changes
if ss.get("gallery_target") != sel_target:
    ss.gallery_n = GRID_COLS * GRID_ROWS
    ss.gallery_target = sel_target
ss.setdefault("gallery_n", GRID_COLS * GRID_ROWS)

# keep the selection valid for this target (gallery curves are valid too)
valid_sel = set(gallery_pool) | set(work["measurement_id"])
if ss.selected not in valid_sel:
    ss.selected = work["measurement_id"].iloc[0] if len(work) else (gallery_pool[0] if gallery_pool else None)

feats = load_all_features(len(df))

# checks + flags over the working set
checks = engine.load_checks()
flags_by_check = {c["name"]: set(engine.flag_dataset(c, feats)) for c in checks}
flags_by_id = {mid: [c["name"] for c in checks if mid in flags_by_check[c["name"]]] for mid in feats}

st.sidebar.markdown("---")
st.sidebar.markdown("**QC checks (skills)**")
active_checks = []
for c in checks:
    on = st.sidebar.checkbox(f"{c['name']}  ·  {len(flags_by_check[c['name']])}", key=f"chk_{c['name']}")
    if on:
        active_checks.append(c["name"])
st.sidebar.caption(f"{len(checks)} checks · {len(feats)} curves loaded · {len(df)} total measurements")

if active_checks:
    keep = set().union(*(flags_by_check[n] for n in active_checks))
    work = work[work["measurement_id"].isin(keep)]


def measurement_row(mid):
    return df[df["measurement_id"] == mid].iloc[0]


def effective_flag(mid):
    """Manual 'problematic' flag: defaults to True when any QC check fires, but the
    user can override either way by clicking the flag cell."""
    return ss.flag_override.get(mid, bool(flags_by_id.get(mid)))


def suggest_negatives(pos_ids, n=4):
    """Contrasting 'normal' curves for the QC Builder: clean binders that pass
    EVERY current check (no flags), with a good staircase, fit and signal -
    preferring the same target(s) as the positives. Feeding these as negatives
    pushes Claude to find the feature that SEPARATES the issue from normal."""
    posset = set(pos_ids)
    pos_targets = {MTARGET.get(m) for m in pos_ids}
    cands = []
    for mid, f in feats.items():
        if mid in posset or not f.get("valid"):
            continue
        if flags_by_id.get(mid):                       # must pass all current checks
            continue
        if not f.get("order_respected"):
            continue
        if (f.get("mean_r2") or 0) < 0.9 or (f.get("snr") or 0) < 15 or (f.get("max_response") or 0) < 5:
            continue
        cands.append(mid)
    same = [m for m in cands if MTARGET.get(m) in pos_targets]
    pool = same if len(same) >= n else same + [m for m in cands if m not in set(same)]
    pool.sort(key=lambda m: -(feats[m].get("mean_r2") or 0))
    return pool[:n]


# ------------------------------------------------------------ short labels
_NICE_LABEL = {
    "unexpected_order": "staircase", "fit_quality": "fit quality",
    "weak_signal": "weak signal", "aggregation": "aggregation",
    "carryover": "carryover", "drift": "drift", "saturation": "saturation",
}


def _check_labels(check_list):
    """Readable per-check column labels (words separated by spaces so headers wrap
    at word boundaries, never mid-word)."""
    used, colmap, labels = set(), {}, []
    for c in check_list:
        cat = (c.get("category") or "").strip()
        base = _NICE_LABEL.get(cat) or (cat.replace("_", " ") if cat else c["name"][:14])
        lbl, i = base, 2
        while lbl in used:
            lbl = f"{base} {i}"; i += 1
        used.add(lbl); colmap[lbl] = c["name"]; labels.append(lbl)
    return labels, colmap


# flagged (failing) cells -> red ✗; passing cells stay blank
_REDX = JsCode("function(p){return p.value ? "
               "{color:'#dc2626',textAlign:'center',fontWeight:'700'} : {textAlign:'center'};}")


# --------------------------------------------------------------------- tabs
tab_review, tab_flags = st.tabs(["\U0001f52c Review", "\U0001f6a9 QC Flags"])

# ============================ REVIEW ============================
with tab_review:
    st.caption(f"**{len(work)}** measurements · target **{sel_target}**" +
               (f" · {len(sel_exps)} experiment(s)" if sel_exps else " · all experiments") +
               (f" · filtered by {', '.join(active_checks)}" if active_checks else ""))

    tbl, viewer = st.columns([0.6, 0.4], gap="medium")
    with tbl:
        t = work.copy()
        t["experiment"] = t["measurement_id"].map(lambda m: MEXP.get(m, "—"))
        t["call"] = t["binder"].map({1: "Binder", 0: "Non-binder"}).fillna("—")
        t["KD (nM)"] = (t["kd"] * 1e9).round(2)
        t["max R"] = t["measurement_id"].map(lambda m: round(feats.get(m, {}).get("max_response") or 0, 1))
        t["reps"] = [rep_counts.get((p, tg), 1) for p, tg in zip(t["protein_id"], t["target"])]
        t["flagged"] = t["measurement_id"].map(effective_flag)   # bool -> checkbox
        flag_cols, colmap = _check_labels(checks)
        for lbl in flag_cols:
            nm = colmap[lbl]
            t[lbl] = t["measurement_id"].map(lambda m, n=nm: "✗" if m in flags_by_check[n] else "")
        view = t[["measurement_id", "name", "target", "experiment", "call", "binding_strength",
                  "KD (nM)", "max R", "reps", "flagged"] + flag_cols] \
            .rename(columns={"binding_strength": "strength"})

        gb = GridOptionsBuilder.from_dataframe(view)
        gb.configure_default_column(sortable=True, filter=False, resizable=True)
        gb.configure_selection("multiple", use_checkbox=True, suppressRowClickSelection=True)
        gb.configure_column("measurement_id", hide=True)
        # minWidth (not just width) stops AgGrid from squishing columns to the
        # container, so every header sits on one line (grid scrolls horizontally)
        gb.configure_column("name", minWidth=190, width=210,
                            checkboxSelection=True, headerCheckboxSelection=True)
        gb.configure_column("target", width=140, minWidth=110, tooltipField="target")
        gb.configure_column("experiment", width=170, minWidth=150, tooltipField="experiment")
        gb.configure_column("call", width=96, minWidth=96)
        gb.configure_column("strength", width=92, minWidth=92)
        gb.configure_column("KD (nM)", type=["numericColumn"], width=92, minWidth=92,
                            comparator=_NULLS_LAST, valueFormatter=_BLANK_NULL)
        gb.configure_column("max R", type=["numericColumn"], width=82, minWidth=82)
        gb.configure_column("reps", type=["numericColumn"], width=66, minWidth=66)
        gb.configure_column("flagged", width=86, minWidth=86, editable=True,
                            cellRenderer="agCheckboxCellRenderer", cellEditor="agCheckboxCellEditor",
                            headerName="🚩 flag",
                            headerTooltip="Manual 'problematic' flag - tick to flag a sample. "
                                          "On by default when any QC check fires; untick to clear.")
        for lbl in flag_cols:
            w = max(74, len(lbl) * 9 + 20)   # fits the whole label on one line
            gb.configure_column(lbl, width=w, minWidth=w, cellStyle=_REDX, headerTooltip=colmap[lbl])
        # inject into the grid iframe: small header font, never break a word
        header_css = {
            ".ag-header-cell-text": {"white-space": "nowrap !important", "font-size": "12px !important"},
            ".ag-header-cell": {"padding-left": "6px !important", "padding-right": "4px !important"},
        }
        # st_aggrid keeps stale data when the dataframe shrinks under a fixed key;
        # remount the grid whenever the filter set changes so the table re-renders
        import hashlib as _hl
        _filt_sig = _hl.md5(repr((sel_target, sel_kind, search, collapse_reps,
                                  tuple(sorted(active_checks)), tuple(sorted(sel_exps)),
                                  len(view))).encode()).hexdigest()[:10]
        grid = AgGrid(view, gridOptions=gb.build(), height=540, theme=AGGRID_THEME,
                      update_on=["selectionChanged", "cellClicked", "cellValueChanged"],
                      custom_css=header_css, fit_columns_on_grid_load=False,
                      allow_unsafe_jscode=True, key=f"aggrid_browse_{_filt_sig}")
        # sync the editable "flagged" checkboxes back into the override store
        gdata = grid.get("data")
        if gdata is not None:
            try:
                for _, _r in gdata.iterrows():
                    _m = _r["measurement_id"]
                    _v = bool(_r["flagged"])
                    if _v != effective_flag(_m):
                        ss.flag_override[_m] = _v
            except Exception:
                pass
        # checkbox multi-select -> QC Builder reference set
        sel = grid.get("selected_rows")
        sel_ids = []
        if isinstance(sel, pd.DataFrame) and not sel.empty:
            sel_ids = sel["measurement_id"].tolist()
        elif isinstance(sel, list):
            sel_ids = [r.get("measurement_id") for r in sel if r]
        sel_ids = [s for s in sel_ids if s]
        # clicking a cell: the "flagged" column toggles the manual flag; any other
        # column loads that curve in the viewer
        ev = grid.get("event_data")
        if ev and ev != ss.get("_prev_grid_event"):
            ss._prev_grid_event = ev
            col = (ev.get("colDef") or {}).get("field") if isinstance(ev, dict) else None
            cm = (ev.get("data") or {}).get("measurement_id") if isinstance(ev, dict) else None
            if cm and col != "flagged":     # the flag checkbox is handled via grid data
                ss.selected = cm

    with viewer:
        mid = ss.selected
        if mid:
            row = measurement_row(mid)
            f = feats.get(mid, {})
            st.markdown(f"#### {short_name(row['name'], 40)}")
            binder = "\U0001f7e2 Binder" if row["binder"] == 1 else ("⚪ Non-binder" if row["binder"] == 0 else "—")
            st.caption(f"`{mid[:12]}` · {binder} · strength {row['binding_strength'] or '—'} · "
                       f"KD {fmt_kd(row['kd'])} · fit {row['fit_model'] or '—'}")
            fl = flags_by_id.get(mid, [])
            st.markdown("**QC flags:** " + (" ".join(f":red[`{n}`]" for n in fl) if fl else ":green[none]"))
            st.plotly_chart(viz.sensorgram(mid, show_fit=True, template=PLOTLY_T),
                            use_container_width=True, key=f"big_{mid}", config={"displaylogo": False})

            def g(k, nd=2):
                v = f.get(k)
                return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else v)
            TIP = {
                "max R": "Maximum binding response across all concentrations (nm). Higher = stronger signal.",
                "SNR": "Signal-to-noise ratio: max response / baseline noise.",
                "mean R2": "Mean fit quality of the 1:1 kinetic model vs the raw trace (1.0 = perfect).",
                "staircase": "Spearman rank correlation of concentration vs response. ~1 = clean staircase.",
                "order ok": "Whether response rises monotonically with concentration (no inversions).",
                "dissoc ret.": "Signal at end of dissociation / end of association. ~1 = barely dissociates (slow off).",
                "assoc compl.": "Fraction of plateau reached at mid-association. Low = slow/creeping on-rate.",
                "drift": "Steepest baseline slope (nm/s). High = baseline drifting.",
            }
            cells = [
                ("max R", g("max_response", 1)), ("SNR", g("snr", 0)), ("mean R2", g("mean_r2")),
                ("staircase", g("spearman_conc_response")),
                ("order ok", "✓" if f.get("order_respected") else "✗"),
                ("dissoc ret.", g("dissoc_retention")), ("assoc compl.", g("assoc_completion")),
                ("drift", g("max_baseline_drift", 3)),
            ]
            row_html = ("<div style='display:flex;justify-content:space-between;gap:6px;"
                        "background:#f4f6f6;border-radius:8px;padding:8px 10px;margin-top:4px'>")
            for lab, val in cells:
                tip = TIP.get(lab, "").replace("'", "&#39;")
                row_html += (f"<div title='{tip}' style='text-align:center;flex:1;cursor:help'>"
                             f"<div style='font-size:0.66rem;color:#64748b'>{lab} "
                             f"<span style='opacity:0.5'>ⓘ</span></div>"
                             f"<div style='font-size:0.9rem;font-weight:600'>{val}</div></div>")
            row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)

    # ---------- QC Builder chatbox (below the table) ----------
    st.divider()
    st.markdown("#### ✨ QC Builder")
    nsel = len(sel_ids)
    st.caption((f"**{nsel}** sample(s) selected as reference issues — Claude will look for the "
                f"common pattern across them." if nsel else
                "Tick samples in the table to use as reference issues, or just describe a filter below."))

    # auto-suggest contrasting "normal" curves as negatives (sharpens the pattern)
    neg_ids = suggest_negatives(sel_ids) if nsel else []
    inc_neg = False
    if neg_ids:
        inc_neg = st.checkbox(
            f"💡 Include {len(neg_ids)} suggested **normal** curves as negative examples "
            f"(contrast — sharpens the pattern & avoids over-broad checks)",
            value=True, key="inc_neg")
        if inc_neg:
            ncols = st.columns(len(neg_ids))
            for c, m in zip(ncols, neg_ids):
                with c:
                    st.image(png_thumb(m, 240, 150))
                    st.caption("✓ normal · " + short_name(PROT_NAME.get(m, ""), 14))

    cc1, cc2 = st.columns([0.86, 0.14])
    nl = cc1.text_input("builder", key="builder_text", label_visibility="collapsed",
                        placeholder="describe your new filter, or select multiple examples as reference issues")
    submit = cc2.button("Submit", type="primary", use_container_width=True,
                        disabled=config.anthropic_key() is None)
    if submit:
        if not nl.strip() and not sel_ids:
            st.warning("Describe a filter or tick example samples in the table.")
        else:
            with st.spinner("Claude is analysing the selected curves (images + data) for a common pattern…"):
                try:
                    pos = [(m, feats.get(m, {}), png_thumb(m, 520, 360)) for m in sel_ids[:8] if m in feats]
                    neg = ([(m, feats.get(m, {}), png_thumb(m, 520, 360)) for m in neg_ids if m in feats]
                           if inc_neg else [])
                    ss.draft = engine.build_check_llm(nl, positives=pos, negatives=neg)
                except Exception as e:
                    st.error(f"Build failed: {e}")

    if ss.draft:
        d = ss.draft
        st.markdown(f"**Pattern found → {d['name']}**")
        if d.get("pattern_summary"):
            st.info(d["pattern_summary"])
        matches = engine.flag_dataset(d, feats)
        st.caption(f"Matches **{len(matches)}** of {len(feats)} measurements across all ProteinBase.")
        with st.expander("check definition"):
            st.write(d.get("description", ""))
            st.caption("Why this works: " + d.get("rationale", ""))
            st.code(json.dumps(d.get("spec", {}), indent=1) if d.get("mode") == "spec"
                    else d.get("code", ""), language="json")
        bb1, bb2 = st.columns([0.32, 0.68])
        if bb1.button("✅ Save & apply as filter", type="primary"):
            engine.save_check(d)
            ss._pending_check = d["name"]      # turn it on as a filter next run
            ss._pending_target = "All targets"  # show matches across all datasets
            ss._pending_exp = []
            ss.draft = None
            st.rerun()
        if bb2.button("Discard"):
            ss.draft = None
            st.rerun()

    # ---------- same-target gallery ----------
    st.divider()
    shown = gallery_pool[:ss.gallery_n]
    glabel = "all targets" if all_targets else sel_target
    st.markdown(f"**Same-target gallery — {glabel}** "
                f"({len(shown)} of {len(gallery_pool)} curves · scroll, **click a plot** to load it above)")
    uris = [thumb_datauri(m) for m in shown]
    titles = [PROT_NAME.get(m, "") for m in shown]
    with st.container(height=470):
        clicked = clickable_images(
            uris, titles=titles,
            div_style={"display": "flex", "flex-wrap": "wrap", "gap": "5px", "justify-content": "flex-start"},
            img_style={"height": "120px", "cursor": "pointer", "border": "1px solid #e5e7eb",
                       "border-radius": "6px"},
            key=f"gallery_{sel_target}_{ss.gallery_n}")
        if clicked is not None and 0 <= clicked < len(shown):
            gmid = shown[clicked]
            if gmid != ss.get("gallery_last_click"):
                ss.gallery_last_click = gmid
                ss.selected = gmid
                st.rerun()
        if ss.gallery_n < len(gallery_pool):
            if st.button(f"⬇ Load 24 more  ({ss.gallery_n}/{len(gallery_pool)})",
                         key="load_more", use_container_width=True):
                ss.gallery_n += GRID_COLS * GRID_ROWS
                st.rerun()

# ============================ QC FLAGS (editable) ============================
with tab_flags:
    st.caption("Every saved QC check (including ones built in QC Builder). Edit names / "
               "categories / descriptions inline; delete a row to remove the check. Each check "
               "is also a sortable column in the Review table.")
    chk_df = pd.DataFrame([{
        "name": c["name"], "category": c.get("category", ""),
        "description": c.get("description", ""), "mode": c.get("mode", "spec"),
        "flagged": len(flags_by_check[c["name"]]), "file": c["_file"],
    } for c in checks])
    edited = st.data_editor(
        chk_df, key="flags_editor", hide_index=True, num_rows="dynamic", use_container_width=True,
        column_config={
            "flagged": st.column_config.NumberColumn("flagged", disabled=True),
            "mode": st.column_config.TextColumn("mode", disabled=True),
            "file": None,
        })

    orig = {c["_file"]: c for c in checks}
    kept = set(edited["file"].dropna()) if "file" in edited.columns else set(orig)
    deleted = set(orig) - kept
    for fn in deleted:
        engine.delete_check_file(fn)
    for _, r in edited.iterrows():
        fn = r.get("file")
        if fn is None or (isinstance(fn, float)):
            continue
        o = orig.get(fn)
        if o and (r["name"] != o["name"] or r["category"] != o.get("category", "")
                  or r["description"] != o.get("description", "")):
            engine.update_check_file(fn, name=r["name"], category=r["category"], description=r["description"])
    if deleted:
        st.rerun()

    st.divider()
    pick = st.selectbox("Inspect flagged curves for", [c["name"] for c in checks])
    if pick:
        ids = sorted(flags_by_check[pick])
        st.caption(f"{len(ids)} flagged across all ProteinBase. Showing up to 12.")
        for r in range(0, min(len(ids), 12), 6):
            cols = st.columns(6)
            for ci, gmid in enumerate(ids[r:r + 6]):
                with cols[ci]:
                    st.image(png_thumb(gmid), use_container_width=True)
                    st.caption(short_name(PROT_NAME.get(gmid, ""), 16))
