"""QC check engine: define, run, and LLM-author checks ("skills").

A check is a small, saveable spec that runs across the dataset and flags
measurements. Two modes:

  spec  - a JSON predicate over the precomputed feature vocabulary (safe,
          the default; what the LLM emits for most criteria)
  code  - a single Python boolean expression over `f` (the feature dict) and
          `np`, for shape-based criteria the feature spec can't express.
          Evaluated with a restricted builtin namespace (escape hatch).

Checks are persisted as YAML skill files in checks/.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import numpy as np
import yaml

import config
import features as featmod

# ---------------------------------------------------------------- spec engine
_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _eval_node(node: dict, f: dict) -> bool:
    if "all" in node:
        return all(_eval_node(n, f) for n in node["all"])
    if "any" in node:
        return any(_eval_node(n, f) for n in node["any"])
    if "not" in node:
        return not _eval_node(node["not"], f)
    feat, op, val = node["feature"], node["op"], node.get("value")
    cur = f.get(feat)
    if cur is None:
        return False
    try:
        return _OPS[op](cur, val)
    except TypeError:
        return False


# --------------------------------------------------------------- code (escape)
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "sum": sum,
    "sorted": sorted, "any": any, "all": all, "range": range,
    "float": float, "int": int, "bool": bool, "round": round, "enumerate": enumerate,
}


def _eval_code(expr: str, f: dict) -> bool:
    return bool(eval(expr, {"__builtins__": _SAFE_BUILTINS, "np": np}, {"f": f}))


def run_check(check: dict, f: dict) -> bool:
    if not f.get("valid"):
        return False
    try:
        if check.get("mode") == "code":
            return _eval_code(check["code"], f)
        return _eval_node(check["spec"], f)
    except Exception:
        return False


# ------------------------------------------------------------- persistence I/O
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48] or "check"


def save_check(check: dict) -> Path:
    check.setdefault("created", _dt.date.today().isoformat())
    name = _slug(check["name"])
    path = config.CHECKS / f"{name}.yaml"
    path.write_text(yaml.safe_dump(check, sort_keys=False), encoding="utf-8")
    return path


def load_checks() -> list[dict]:
    out = []
    for p in sorted(config.CHECKS.glob("*.yaml")):
        try:
            c = yaml.safe_load(p.read_text(encoding="utf-8"))
            c["_file"] = p.name
            out.append(c)
        except Exception:
            continue
    return out


def delete_check(name: str):
    p = config.CHECKS / f"{_slug(name)}.yaml"
    if p.exists():
        p.unlink()


def delete_check_file(filename: str):
    p = config.CHECKS / filename
    if p.exists():
        p.unlink()


def update_check_file(filename: str, **fields):
    """Edit a saved check in place (keeps the file stable across name edits)."""
    p = config.CHECKS / filename
    if not p.exists():
        return
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    c.update({k: v for k, v in fields.items() if v is not None})
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")


# ------------------------------------------------------------------ LLM author
SPEC_GRAMMAR = """\
A spec is a predicate tree. Emit EXACTLY these node shapes and no others:
- Leaf comparison: {"feature": "<feature_name>", "op": "<one of: < <= > >= == !=>", "value": <number | true | false>}
- AND:  {"all": [ <node>, <node>, ... ]}
- OR:   {"any": [ <node>, ... ]}
- NOT:  {"not": <node> }

Worked example - "broken staircase (response order doesn't track concentration)":
{"any": [
  {"feature": "order_respected", "op": "==", "value": false},
  {"feature": "spearman_conc_response", "op": "<", "value": 0.9}
]}

Use ONLY the keys: feature, op, value, all, any, not. Do NOT invent other keys
(no "eq", "and", "or", "field", "lhs", "operator", and do not nest a feature
inside another object). value must be a plain number or boolean."""

SYSTEM = """You are a QC-check author for protein-binding (SPR/BLI) sensorgram data.
A scientist describes, in plain language or via tagged examples, a data-quality
criterion they want to flag across a dataset. You turn it into a concrete,
runnable check over a fixed feature vocabulary.

HOW YOU SEE EACH CURVE: for every measurement referenced you are given BOTH
(a) a rendered sensorgram IMAGE (time on x, response on y, one line per analyte
concentration, dotted line = the 1:1 kinetic fit), and (b) its UNDERLYING DATA
as a precomputed numeric feature vector. Use the image to recognise the SHAPE of
the pattern the scientist means; express the actual check in terms of the numeric
features, because at run time the check executes ONLY on those features across
the whole dataset (the image is not available then). If the pattern is visible in
the image but not separable by the listed features, say so in the rationale and
use mode="code" over `f`.

%s

Adaptyv's existing QC vocabulary (use as category when it fits): staircase,
carryover, weak_signal, drift, saturation, low_loading, unexpected_order.

%s

When the scientist gives EXAMPLE curves, your job is to act like an analyst: look
across all of them (images + numeric features), find what they share and how they
differ from normal curves, state that common pattern in `pattern_summary`, then
encode it as a check whose thresholds actually separate the examples. This is how
a reusable detector is discovered from a handful of flagged samples.

STRICTNESS: prefer FALSE NEGATIVES over false positives. A QC check should flag
only a small, unambiguous minority (aim well under ~10%% of the dataset). Be
conservative:
- If the criterion lists multiple conditions ("X and Y", "A with B"), require ALL
  of them with `all`. Do NOT use `any` (OR) unless the conditions are genuinely
  interchangeable ways of saying the SAME thing. Two different physical phenomena
  joined by "and" must both be required.
- Shape features (assoc_completion, dissoc_retention, carryover_frac, staircase
  metrics) are MEANINGLESS on weak/noisy curves and take wild, unphysical values
  there. Whenever the criterion is about curve SHAPE, you MUST also require real
  binding signal in the SAME `all` block: order_respected == true AND
  max_response above a clear value (e.g. > 8) AND/OR snr > 12. This stops the
  check firing on noise.
- Put thresholds at the EXTREME TAIL of the distribution (appended below at the
  end of this prompt), e.g. the 5th/95th percentile of real-signal curves, not
  near the median. "really slow", "very", "clear" => push to the tail.

Return a check via the emit_check tool. Prefer mode="spec". Only use mode="code"
(a single Python boolean expression over `f` the feature dict and `np`) when the
criterion truly cannot be expressed as a spec. A measurement is FLAGGED when the
check evaluates True. Explain your reasoning in `rationale`.""" % (
    featmod.FEATURE_GLOSSARY, SPEC_GRAMMAR)

_TOOL = {
    "name": "emit_check",
    "description": "Emit a concrete QC check.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "short human name, e.g. 'Staircase violation'"},
            "category": {"type": "string"},
            "description": {"type": "string", "description": "one line, what it flags"},
            "mode": {"type": "string", "enum": ["spec", "code"]},
            "spec": {"type": "object", "description": "predicate tree (when mode=spec)"},
            "code": {"type": "string", "description": "python bool expr over f, np (when mode=code)"},
            "rationale": {"type": "string", "description": "why this captures the described pattern"},
            "pattern_summary": {"type": "string", "description":
                "2-4 sentence plain-language description of the COMMON PATTERN you see across the "
                "provided example curves (shape in the image + what the features show), written for "
                "a scientist. If no examples were given, summarise what the criterion targets."},
        },
        "required": ["name", "category", "description", "mode", "rationale", "pattern_summary"],
    },
}


_VALID_FEATURES = {
    "n_concentrations", "concentrations", "plateau_by_conc", "max_response",
    "noise", "snr", "min_r2", "mean_r2", "spearman_conc_response",
    "n_order_inversions", "order_respected", "saturation_top_step_frac",
    "carryover_frac", "dissoc_retention", "assoc_completion",
    "max_baseline_drift", "weak_signal", "fit_model", "kon", "koff", "kd",
}


def validate_spec(node, path="spec") -> str | None:
    """Return an error string if the spec node is malformed, else None."""
    if not isinstance(node, dict):
        return f"{path}: expected an object, got {type(node).__name__}"
    keys = set(node)
    if keys & {"all", "any"}:
        key = "all" if "all" in node else "any"
        if not isinstance(node[key], list) or not node[key]:
            return f"{path}.{key}: must be a non-empty list"
        for i, child in enumerate(node[key]):
            err = validate_spec(child, f"{path}.{key}[{i}]")
            if err:
                return err
        return None
    if "not" in node:
        return validate_spec(node["not"], f"{path}.not")
    if keys == {"feature", "op", "value"} or keys == {"feature", "op"}:
        if node["feature"] not in _VALID_FEATURES:
            return f"{path}: unknown feature '{node['feature']}'"
        if node["op"] not in _OPS:
            return f"{path}: unknown op '{node['op']}'"
        return None
    return (f"{path}: invalid node keys {sorted(keys)}. Use a leaf "
            "{feature,op,value} or a combinator {all|any|not}.")


import base64

# the exact numeric "underlying data" representation the LLM receives per curve
PAYLOAD_KEYS = [
    "n_concentrations", "concentrations", "plateau_by_conc", "max_response",
    "noise", "snr", "min_r2", "mean_r2", "spearman_conc_response",
    "n_order_inversions", "order_respected", "saturation_top_step_frac",
    "carryover_frac", "dissoc_retention", "assoc_completion",
    "max_baseline_drift", "weak_signal", "fit_model", "kon", "koff", "kd",
]


def curve_payload(feats: dict) -> dict:
    """The structured data the model sees for a curve (also shown to the user)."""
    out = {}
    for k in PAYLOAD_KEYS:
        v = feats.get(k)
        if isinstance(v, float):
            v = round(v, 4)
        elif isinstance(v, list):
            v = [round(x, 3) if isinstance(x, float) else x for x in v]
        out[k] = v
    return out


def _curve_content(label: str, mid: str, feats: dict, png: bytes | None) -> list:
    """A multimodal block pair: the numeric data + the rendered image of one curve."""
    blocks = [{"type": "text", "text":
               f"{label} (measurement {mid}).\n"
               f"Underlying data — precomputed numeric features (these are what the "
               f"check executes on):\n{json.dumps(curve_payload(feats))}"}]
    if png:
        blocks.append({"type": "text", "text":
                       "Rendered sensorgram image of this same measurement "
                       "(x=time, y=response; one line per concentration, dotted = 1:1 fit):"})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(png).decode()}})
    return blocks


def _client():
    import anthropic
    key = config.anthropic_key()
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY found (checked env + WSL .env).")
    return anthropic.Anthropic(api_key=key)


def _emit(client, messages, system, model=None):
    """One tool-forced call -> (resp, check dict), with a spec-grammar repair round."""
    def call(msgs):
        resp = client.messages.create(
            model=model or config.ANTHROPIC_MODEL, max_tokens=1500, system=system,
            tools=[_TOOL], tool_choice={"type": "tool", "name": "emit_check"}, messages=msgs)
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        return resp, block

    resp, block = call(messages)
    check = dict(block.input)
    if check.get("mode") == "spec" and validate_spec(check.get("spec", {})):
        err = validate_spec(check.get("spec", {}))
        messages = messages + [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id,
             "is_error": True, "content": f"The spec was rejected: {err}\n\nRe-emit using "
             f"ONLY this grammar:\n{SPEC_GRAMMAR}"}]}]
        _, block = call(messages)
        check = dict(block.input)
    return check


def build_check_llm(nl: str, current=None, positives=None, negatives=None, model=None,
                    feature_stats: str = "") -> dict:
    """Author a check with Claude.

    current/positives/negatives : (mid, feats, png_bytes) - shown as image + data.
    feature_stats : text block of feature distribution percentiles (for tail thresholds).
    """
    client = _client()
    content = [{"type": "text", "text":
                f"Criterion: {nl.strip() or '(infer the common pattern from the tagged examples)'}"}]
    if current:
        content += _curve_content("The curve the scientist is currently viewing", *current)
    for c in (positives or []):
        content += _curve_content("TAGGED example that SHOULD be flagged", *c)
    for c in (negatives or []):
        content += _curve_content("Example that should NOT be flagged", *c)

    system = SYSTEM + ("\n\n" + feature_stats if feature_stats else "")
    check = _emit(client, [{"role": "user", "content": content}], system, model)
    check["source"] = "llm"
    check["nl"] = nl
    if positives:
        check["examples"] = [c[0] for c in positives]
    return check


def tighten_check_llm(check: dict, flagged_n: int, total: int, target_frac: float = 0.08,
                      feature_stats: str = "", model=None) -> dict:
    """Ask Claude to make an over-broad check stricter (fewer false positives)."""
    client = _client()
    spec = json.dumps(check.get("spec")) if check.get("mode") == "spec" else check.get("code")
    msg = (f"This check flags {flagged_n} of {total} measurements "
           f"({100*flagged_n/max(total,1):.0f}%), which is TOO BROAD for a QC flag.\n"
           f"Criterion it targets: {check.get('nl') or check.get('description','')}\n"
           f"Current {check.get('mode')}: {spec}\n\n"
           f"Re-emit a STRICTER version that flags well under {int(target_frac*100)}% — "
           f"prefer false negatives over false positives. Require ALL conditions with `all` "
           f"(no `any`/OR across different phenomena), push thresholds to the extreme tail, and "
           f"require real binding signal (order_respected, max_response, snr) so shape features "
           f"aren't evaluated on noise. Keep the same name/category.")
    system = SYSTEM + ("\n\n" + feature_stats if feature_stats else "")
    out = _emit(client, [{"role": "user", "content": msg}], system, model)
    out["source"] = "llm"
    out["nl"] = check.get("nl", "")
    if check.get("examples"):
        out["examples"] = check["examples"]
    return out


# ------------------------------------------------------- run across a dataset
def flag_dataset(check: dict, feats_by_id: dict[str, dict]) -> list[str]:
    """Return the measurement ids flagged by this check."""
    return [mid for mid, f in feats_by_id.items() if run_check(check, f)]
