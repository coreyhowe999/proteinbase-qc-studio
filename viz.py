"""Sensorgram plotting (Plotly). Reused in Browse, Context, and the builder preview."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import config
import features as featmod

# low->high concentration colour ramp (teal -> ink), matches the docs sensorgram
_RAMP = ["#8fd9cf", "#5cc8bb", "#2bb6a6", "#14a89b", "#0d9488", "#0f766e", "#134e4a"]


def _fit_xy(series):
    fit = None
    if isinstance(series.get("fit"), dict) and series["fit"].get("association"):
        fit = series["fit"]
    elif isinstance(series.get("fits"), dict) and series["fits"]:
        fit = series["fits"].get("standard") or next(iter(series["fits"].values()))
    if not fit:
        return None
    t, y = [], []
    for ph in ("association", "dissociation"):
        p = fit.get(ph) or {}
        if p.get("t") and len(p["t"]) == len(p.get("y", [])):
            t += list(p["t"]); y += list(p["y"])
    if not t:
        return None
    order = np.argsort(t)
    return np.array(t)[order], np.array(y)[order]


def sensorgram(measurement_id: str, title: str = "", show_fit: bool = True,
               mini: bool = False, template: str = "plotly_white") -> go.Figure:
    curve = featmod.load_curve(measurement_id)
    series = sorted(curve.values(), key=lambda s: s.get("concentration") or 0)
    fig = go.Figure()
    n = len(series)
    for i, s in enumerate(series):
        raw = s.get("raw") or {}
        t, y = raw.get("t") or [], raw.get("y") or []
        col = _RAMP[int(i / max(n - 1, 1) * (len(_RAMP) - 1))]
        conc = s.get("concentration")
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", line=dict(color=col, width=1.6),
            name=f"{conc:g} nM" if conc else "trace",
            hovertemplate="t=%{x:.0f}s<br>R=%{y:.2f} nm<extra></extra>",
            showlegend=not mini))
        if show_fit and not mini:
            fx = _fit_xy(s)
            if fx is not None:
                fig.add_trace(go.Scatter(
                    x=fx[0], y=fx[1], mode="lines",
                    line=dict(color=col, width=2.2, dash="dot"),
                    name=f"fit {conc:g} nM", showlegend=False, opacity=0.55,
                    hoverinfo="skip"))
    layout = dict(
        template=template,
        # extra top margin keeps the floating modebar clear of the plot;
        # legend sits BELOW the plot so it never collides with the toolbar
        margin=dict(l=8, r=8, t=34, b=8),
        height=180 if mini else 400,
        xaxis_title=None if mini else "Time (s)",
        yaxis_title=None if mini else "Response (nm)",
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="top", y=-0.18, x=0.5, xanchor="center"),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13))
    fig.update_layout(**layout)
    if not title:                       # avoid an empty/"undefined" title slot
        fig.update_layout(title_text="")
    if mini:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    return fig


def sensorgram_png(measurement_id: str, title: str = "", width: int = 560,
                   height: int = 380, dark: bool = False) -> bytes:
    """Static PNG of a sensorgram via matplotlib (kaleido is unreliable on this box).
    This is the exact image handed to the LLM in the check builder."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = "#161a23" if dark else "white"
    fg = "#d8dde6" if dark else "#334155"
    curve = featmod.load_curve(measurement_id)
    series = sorted(curve.values(), key=lambda s: s.get("concentration") or 0)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    n = len(series)
    for i, s in enumerate(series):
        raw = s.get("raw") or {}
        col = _RAMP[int(i / max(n - 1, 1) * (len(_RAMP) - 1))]
        conc = s.get("concentration")
        ax.plot(raw.get("t") or [], raw.get("y") or [], color=col, lw=1.1,
                label=f"{conc:g} nM" if conc else "trace")
        fx = _fit_xy(s)
        if fx is not None:
            ax.plot(fx[0], fx[1], color=col, lw=1.4, ls=":", alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=8, color=fg)
    ax.set_ylabel("Response (nm)", fontsize=8, color=fg)
    if title:
        ax.set_title(title, fontsize=9, color=fg)
    ax.tick_params(labelsize=7, colors=fg)
    leg = ax.legend(fontsize=6, loc="upper right", framealpha=0.5)
    for txt in leg.get_texts():
        txt.set_color(fg)
    for sp in ax.spines.values():
        sp.set_color(fg)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


def staircase_plot(feats: dict, template: str = "plotly_white") -> go.Figure:
    """Plateau response vs concentration - the staircase view used for QC."""
    concs = feats.get("concentrations") or []
    plat = feats.get("plateau_by_conc") or []
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=concs, y=plat, mode="lines+markers",
        line=dict(color="#0d9488", width=2),
        marker=dict(size=9, color="#0f766e")))
    fig.update_layout(
        template=template, height=260,
        margin=dict(l=8, r=8, t=30, b=8),
        title=dict(text="Response vs concentration (staircase)", font=dict(size=12)),
        xaxis_title="Concentration (nM)", yaxis_title="Plateau response (nm)",
        xaxis_type="log")
    return fig
