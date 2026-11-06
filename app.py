import io
import os
import json
import math
import uuid
import tempfile
import hashlib
import datetime as dt
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import av
except Exception:
    av = None

st.set_page_config(
    page_title="Autonomous Ad Revenue Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Autonomous Ad Revenue Optimizer"
APP_VERSION = "0.2.0-synthetic-prototype"
MODEL_FAMILY = "hgb-bid-ensemble-v2"

SYNTHETIC_NOTICE = (
    "All data in this demo are synthetic. No real user data, no paid APIs, "
    "no live ad-server writes. Revenue uplift is a simulated paired counterfactual, "
    "not a commercial guarantee."
)

CORE_COLUMNS = [
    "impression_uid",
    "event_time",
    "publisher_id",
    "ad_unit",
    "country",
    "device",
    "audience_segment",
    "demand_source",
    "consent_state",
    "fraud_flag",
    "bid_floor_cpm",
    "max_bid_cpm",
    "second_bid_cpm",
    "winning_bid_cpm",
    "revenue_cpm",
    "cleared",
    "latency_ms",
    "segment_key",
    "model_version",
    "experiment_arm",
    "decision_reason",
    "approval_status",
]

AUX_COLUMNS = [
    "predicted_max_bid_cpm",
    "prediction_uncertainty_cpm",
    "baseline_revenue_cpm",
]

TEMP_COLUMNS = [
    "_fallback",
    "_rate_limited",
    "_exploration",
    "_pending",
    "_approved",
]

ALL_COLUMNS = CORE_COLUMNS + AUX_COLUMNS

COUNTRIES = ["US", "CA", "UK", "DE", "FR", "JP", "BR", "IN", "MX", "AU"]
COUNTRY_VALUE = {
    "US": 1.00,
    "CA": 0.84,
    "UK": 0.91,
    "DE": 0.89,
    "FR": 0.83,
    "JP": 0.96,
    "BR": 0.47,
    "IN": 0.36,
    "MX": 0.49,
    "AU": 0.88,
}
COUNTRY_PROB = np.array([0.22, 0.08, 0.10, 0.09, 0.08, 0.10, 0.07, 0.09, 0.06, 0.11], dtype=float)
COUNTRY_PROB /= COUNTRY_PROB.sum()

DEVICES = ["mobile", "desktop", "tablet", "ctv"]
DEVICE_VALUE = {"mobile": 0.96, "desktop": 1.02, "tablet": 0.82, "ctv": 1.24}
DEVICE_PROB = np.array([0.48, 0.26, 0.13, 0.13], dtype=float)
DEVICE_PROB /= DEVICE_PROB.sum()

SEGMENTS = ["news", "sports", "entertainment", "finance", "health", "retail"]
SEGMENT_VALUE = {
    "news": 0.95,
    "sports": 1.06,
    "entertainment": 1.00,
    "finance": 1.21,
    "health": 1.10,
    "retail": 0.98,
}
SEGMENT_PROB = np.array([0.20, 0.18, 0.18, 0.16, 0.14, 0.14], dtype=float)
SEGMENT_PROB /= SEGMENT_PROB.sum()

DEMAND_SOURCES = ["direct", "dsp_alpha", "dsp_beta", "dsp_gamma", "contextual"]
DEMAND_VALUE = {
    "direct": 1.16,
    "dsp_alpha": 1.02,
    "dsp_beta": 0.93,
    "dsp_gamma": 0.88,
    "contextual": 0.76,
}
DEMAND_PROB = np.array([0.18, 0.25, 0.21, 0.16, 0.20], dtype=float)
DEMAND_PROB /= DEMAND_PROB.sum()

PUBLISHERS = ["pub_newsnet", "pub_sportsplus", "pub_finance_daily", "pub_entertain_hub"]
AD_UNITS = ["top_banner", "mid_article", "sidebar", "video_pre_roll"]
EXPERIMENT_ARMS = ["control", "treat_a", "treat_b"]

COUNTRY_ENC = {c: i + 1 for i, c in enumerate(COUNTRIES)}
DEVICE_ENC = {d: i + 1 for i, d in enumerate(DEVICES)}
SEGMENT_ENC = {s: i + 1 for i, s in enumerate(SEGMENTS)}
SEGMENT_ENC["contextual_unknown"] = 0
SEGMENT_ENC["unknown"] = 0
DEMAND_ENC = {d: i + 1 for i, d in enumerate(DEMAND_SOURCES)}
DEMAND_ENC["unknown"] = 0

SCENARIOS = [
    "normal",
    "low_data",
    "fraud_spike",
    "schema_drift",
    "consent_drop",
    "model_drift",
    "high_volatility",
]

POLICIES = ["static", "quantile", "model", "autonomous"]
RISK_MODES = ["conservative", "balanced", "aggressive"]

SCENARIO_DESCRIPTIONS = {
    "normal": "Stable synthetic auction traffic with normal consent, fraud, and schema behavior.",
    "low_data": "Sparse traffic and limited segment history. The system should prefer conservative fallback.",
    "fraud_spike": "Invalid traffic surges. Fraud rows should be excluded from training and revenue.",
    "schema_drift": "Upstream adapter intermittently drops demand_source. Validator should detect drift and degrade safely.",
    "consent_drop": "More no-consent traffic. System should use contextual-only segment keys and avoid personalization assumptions.",
    "model_drift": "Bid-value relationships shift midstream. Drift monitor should trigger conservative behavior.",
    "high_volatility": "Bid distribution becomes noisy. Uncertainty penalties and guardrails should matter more.",
}

FIELD_DICTIONARY = [
    {"field": "impression_uid", "type": "string", "unit": "UUID", "description": "Unique auction impression identifier.", "example": "7c8f9d2e-..."},
    {"field": "event_time", "type": "timestamp", "unit": "UTC ISO-8601", "description": "Auction event time.", "example": "2026-11-06T14:03:21+00:00"},
    {"field": "publisher_id", "type": "string", "unit": "ID", "description": "Publisher or property identifier.", "example": "pub_newsnet"},
    {"field": "ad_unit", "type": "string", "unit": "ID", "description": "Placement identifier.", "example": "mid_article"},
    {"field": "country", "type": "string", "unit": "ISO-2", "description": "Geography.", "example": "US"},
    {"field": "device", "type": "string", "unit": "enum", "description": "Device type.", "example": "mobile"},
    {"field": "audience_segment", "type": "string", "unit": "enum", "description": "Contextual or consented audience segment.", "example": "finance"},
    {"field": "demand_source", "type": "string", "unit": "enum", "description": "Demand channel or DSP.", "example": "dsp_alpha"},
    {"field": "consent_state", "type": "boolean", "unit": "true/false", "description": "Whether personalized signal is permitted.", "example": True},
    {"field": "fraud_flag", "type": "boolean", "unit": "true/false", "description": "Invalid traffic indicator.", "example": False},
    {"field": "bid_floor_cpm", "type": "float", "unit": "USD CPM", "description": "Applied floor price.", "example": 6.12},
    {"field": "max_bid_cpm", "type": "float", "unit": "USD CPM", "description": "Synthetic latent highest bid.", "example": 8.44},
    {"field": "second_bid_cpm", "type": "float", "unit": "USD CPM", "description": "Synthetic latent second price.", "example": 6.02},
    {"field": "winning_bid_cpm", "type": "float", "unit": "USD CPM", "description": "Paid price after floor logic.", "example": 6.12},
    {"field": "revenue_cpm", "type": "float", "unit": "USD CPM", "description": "Revenue per 1000 impressions.", "example": 6.12},
    {"field": "cleared", "type": "boolean", "unit": "true/false", "description": "Whether auction cleared.", "example": True},
    {"field": "latency_ms", "type": "float", "unit": "milliseconds", "description": "Auction latency.", "example": 21.4},
    {"field": "segment_key", "type": "string", "unit": "composite", "description": "Country|device|segment|demand or contextual fallback.", "example": "US|mobile|finance|dsp_alpha"},
    {"field": "predicted_max_bid_cpm", "type": "float", "unit": "USD CPM", "description": "Model estimate of highest bid.", "example": 8.10},
    {"field": "prediction_uncertainty_cpm", "type": "float", "unit": "USD CPM", "description": "Model uncertainty estimate.", "example": 0.95},
    {"field": "decision_reason", "type": "string", "unit": "text", "description": "Explainability reason for action.", "example": "expected_revenue_optimizer; rate_limited"},
    {"field": "approval_status", "type": "string", "unit": "enum", "description": "not_required, pending_human_approval, simulated_human_approved, auto_within_guardrail.", "example": "auto_within_guardrail"},
    {"field": "baseline_revenue_cpm", "type": "float", "unit": "USD CPM", "description": "Paired counterfactual static-baseline revenue.", "example": 4.00},
    {"field": "model_version", "type": "string", "unit": "ID", "description": "Traceable model version.", "example": "hgb-bid-ensemble-v2-0.412"},
    {"field": "experiment_arm", "type": "string", "unit": "ID", "description": "Control or treatment arm.", "example": "treat_a"},
]

HANDLING_RULES = [
    {"condition": "Missing required column", "handling": "Add column, mark schema drift, reduce quality score, use fallback if severe."},
    {"condition": "Duplicate impression_uid", "handling": "Keep first occurrence, count duplicate, reject later duplicate."},
    {"condition": "Malformed timestamp", "handling": "Coerce to UTC, fill with watermark, count malformed."},
    {"condition": "Negative revenue or latency", "handling": "Take absolute value or quarantine, count negative values."},
    {"condition": "Late-arriving record", "handling": "Flag if earlier than watermark minus tolerance."},
    {"condition": "Outlier bid or revenue", "handling": "Clip to plausible bounds and flag outlier."},
    {"condition": "Missing demand_source", "handling": "Encode as unknown and reduce confidence."},
    {"condition": "No consent", "handling": "Use contextual-only segment key and avoid personalization assumptions."},
    {"condition": "Fraud flag true", "handling": "Exclude from training and set revenue to zero."},
    {"condition": "Low data quality", "handling": "Use deterministic fallback floor."},
    {"condition": "High model uncertainty", "handling": "Penalize aggressive floors or fallback in conservative mode."},
]

RISK_REGISTER = [
    {
        "risk": "Data availability failure",
        "project_example": "Upstream adapter drops demand_source for US mobile finance traffic.",
        "mitigation": "Schema validator adds unknown, lowers quality score, triggers conservative fallback.",
        "human_approval_required": "No for detection; yes for disabling validation.",
    },
    {
        "risk": "Model error",
        "project_example": "Bid-value model overestimates finance CPM during holiday demand shift.",
        "mitigation": "Uncertainty penalty, chronological validation, drift monitor, fallback to quantile/static.",
        "human_approval_required": "Yes for large floor changes above threshold.",
    },
    {
        "risk": "Privacy misuse",
        "project_example": "System attempts to use sensitive audience signal without consent.",
        "mitigation": "Consent-aware segment key, contextual-only fallback, no real PII in demo.",
        "human_approval_required": "Yes for adding new targeting signals.",
    },
    {
        "risk": "Fairness / discriminatory pricing",
        "project_example": "Floor differs sharply across protected-proxy segments without business justification.",
        "mitigation": "Subgroup monitoring, prohibited category policy, explainability review.",
        "human_approval_required": "Yes for expanding segment definitions.",
    },
    {
        "risk": "Operational failure",
        "project_example": "Pricing service outage or model service latency spike.",
        "mitigation": "Kill switch, static fallback, circuit breaker, audit log.",
        "human_approval_required": "Yes for disabling fallback.",
    },
    {
        "risk": "Adoption resistance",
        "project_example": "Yield managers distrust autonomous recommendations.",
        "mitigation": "Shadow mode, evidence inspector, approval queue, simulated before live rollout.",
        "human_approval_required": "Yes for production autonomy.",
    },
    {
        "risk": "Cost overrun",
        "project_example": "Retraining too frequently on high-cardinality segments.",
        "mitigation": "Batch retraining, caching, modular monolith first, split services only with evidence.",
        "human_approval_required": "No.",
    },
]

ACCEPTANCE_CHECKS = [
    {"area": "Launch", "check": "Fresh developer can run streamlit run app.py and see a non-blank dashboard immediately."},
    {"area": "Demo video", "check": "App auto-generates MP4 or GIF fallback, or at minimum a final PNG frame if encoding fails."},
    {"area": "Synthetic data", "check": "All data are labeled synthetic and no API keys are required."},
    {"area": "Baseline", "check": "Static baseline policy is available and comparable to optimizer policy."},
    {"area": "Guardrails", "check": "Floor bounds, rate limits, fill-rate target, fraud exclusion, and approval threshold are enforced."},
    {"area": "Failure handling", "check": "low_data, fraud_spike, schema_drift, consent_drop, and model_drift do not crash the app."},
    {"area": "Auditability", "check": "Decisions, fallbacks, approvals, and drift events appear in audit history."},
    {"area": "Explainability", "check": "Each recommendation includes decision_reason and evidence fields."},
    {"area": "Evaluation", "check": "Model tab shows MAE, RMSE, R2 if available, coverage, and subgroup error."},
    {"area": "Business case", "check": "Calculator uses user-entered assumptions and does not fabricate customer results."},
]

LICENSE_INVENTORY = [
    {"component": "Python", "license": "PSF-like", "note": "OK for local prototype."},
    {"component": "Streamlit", "license": "Apache-2.0", "note": "OK."},
    {"component": "pandas", "license": "BSD-3-Clause", "note": "OK."},
    {"component": "NumPy", "license": "BSD-3-Clause", "note": "OK."},
    {"component": "scikit-learn", "license": "BSD-3-Clause", "note": "OK."},
    {"component": "Pydantic", "license": "MIT", "note": "OK."},
    {"component": "Pillow", "license": "MIT-CMU / HPND variant", "note": "OK."},
    {"component": "PyAV", "license": "BSD", "note": "Bundled FFmpeg components may be LGPL/GPL depending build; verify before commercial redistribution."},
    {"component": "Synthetic data", "license": "Generated locally", "note": "No third-party data license issue."},
]

ROADMAP = [
    {"milestone": "M0", "deliverable": "Product definition, synthetic data contract, risk register."},
    {"milestone": "M1", "deliverable": "Runnable Streamlit prototype with auto-demo video."},
    {"milestone": "M2", "deliverable": "Static, quantile, model, and autonomous policies."},
    {"milestone": "M3", "deliverable": "Guardrails, approvals, audit trail, failure scenarios."},
    {"milestone": "M4", "deliverable": "Evaluation dashboard, subgroup checks, business calculator."},
    {"milestone": "M5", "deliverable": "Packaging, tests, docs, license inventory."},
    {"milestone": "M6", "deliverable": "Pilot design with human oversight and controlled experiment."},
]


class OptimizerConfig(BaseModel):
    seed: int = 42
    duration_sec: int = 6
    fps: int = 10
    batch_size: int = 70
    history_size: int = 2200
    base_floor_cpm: float = 4.0
    min_floor_cpm: float = 0.75
    max_floor_cpm: float = 28.0
    max_change_pct: float = 25.0
    exploration_rate: float = 0.05
    min_fill_rate: float = 0.55
    require_approval_above_pct: float = 12.0
    risk_mode: str = "balanced"
    policy: str = "autonomous"
    scenario: str = "normal"
    simulate_human_approval: bool = True


def as_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


def replace_config(config: OptimizerConfig, **updates: Any) -> OptimizerConfig:
    data = as_dict(config)
    data.update(updates)
    return OptimizerConfig(**data)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def finite_or_nan(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def fmt_usd(x: Any) -> str:
    return f"${safe_float(x, 0.0):,.2f}"


def fmt_pct(x: Any) -> str:
    return f"{safe_float(x, 0.0) * 100:.1f}%"


def fmt_num(x: Any, digits: int = 2) -> str:
    return f"{safe_float(x, 0.0):,.{digits}f}"


def jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, default=jsonable, indent=2).encode("utf-8")


def unique_strings(series: pd.Series) -> List[str]:
    vals = [str(x) for x in series.dropna().unique() if str(x) not in ("", "nan", "None")]
    return sorted(set(vals))


def segment_key_from_row(country: Any, device: Any, segment: Any, demand: Any, consent: Any) -> str:
    c = str(country) if country not in (None, "") else "unknown"
    d = str(device) if device not in (None, "") else "unknown"
    cons = bool(consent)
    seg = str(segment) if cons and segment not in (None, "") else "contextual_unknown"
    if demand is None or (isinstance(demand, float) and math.isnan(demand)) or str(demand).strip() == "":
        dem = "unknown"
    else:
        dem = str(demand)
    return f"{c}|{d}|{seg}|{dem}"


_FONT_CACHE: Dict[int, Any] = {}


def get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for name in ("arial.ttf", "arialbd.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            f = ImageFont.truetype(name, size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int = 12, fill=None, outline=None, width: int = 1):
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        draw.rectangle(xy, fill=fill, outline=outline)


def draw_card(draw: ImageDraw.ImageDraw, xy, title: str, value: str, subtitle: str = "", accent=(34, 197, 94)):
    rounded_rect(draw, xy, radius=14, fill=(17, 24, 39), outline=(51, 65, 85), width=2)
    x1, y1, _, _ = xy
    draw.text((x1 + 14, y1 + 12), title[:34], font=get_font(14), fill=(148, 163, 184))
    draw.text((x1 + 14, y1 + 34), value[:24], font=get_font(24), fill=(248, 250, 252))
    if subtitle:
        draw.text((x1 + 14, y1 + 70), subtitle[:42], font=get_font(12), fill=accent)


def draw_series_chart(draw: ImageDraw.ImageDraw, box, rows: List[Dict[str, Any]], series, title: str):
    x1, y1, x2, y2 = box
    rounded_rect(draw, box, radius=14, fill=(17, 24, 39), outline=(51, 65, 85), width=2)
    draw.text((x1 + 14, y1 + 10), title, font=get_font(14), fill=(226, 232, 240))

    px1, py1, px2, py2 = x1 + 58, y1 + 38, x2 - 20, y2 - 32
    draw.rectangle([px1, py1, px2, py2], fill=(2, 6, 23), outline=(51, 65, 85))

    if not rows:
        return

    all_vals: List[float] = []
    for key, _, _ in series:
        vals = [safe_float(r.get(key, 0.0)) for r in rows]
        all_vals.extend(vals)

    if not all_vals:
        return

    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0

    for key, color, label in series:
        vals = [safe_float(r.get(key, 0.0)) for r in rows]
        pts = []
        for i, v in enumerate(vals):
            x = px1 + (px2 - px1) * (i / max(len(vals) - 1, 1))
            y = py2 - (py2 - py1) * ((v - vmin) / (vmax - vmin))
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        for p in pts:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)

    lx, ly = px1 + 8, py1 + 8
    for _, color, label in series:
        draw.line([(lx, ly + 6), (lx + 18, ly + 6)], fill=color, width=4)
        draw.text((lx + 24, ly), label, font=get_font(12), fill=(203, 213, 225))
        ly += 18

    draw.text((px1 - 52, py2 - 8), f"{vmin:,.1f}", font=get_font(11), fill=(148, 163, 184))
    draw.text((px1 - 52, py1), f"{vmax:,.1f}", font=get_font(11), fill=(148, 163, 184))


def draw_hbar(draw: ImageDraw.ImageDraw, box, items: List[Tuple[str, float]], title: str):
    x1, y1, x2, y2 = box
    rounded_rect(draw, box, radius=14, fill=(17, 24, 39), outline=(51, 65, 85), width=2)
    draw.text((x1 + 14, y1 + 10), title, font=get_font(14), fill=(226, 232, 240))

    if not items:
        draw.text((x1 + 14, y1 + 40), "No segment data yet.", font=get_font(12), fill=(148, 163, 184))
        return

    max_abs = max(abs(v) for _, v in items) or 1.0
    bar_x1 = x1 + 170
    bar_x2 = x2 - 80
    row_h = 24

    for i, (label, val) in enumerate(items[:5]):
        y = y1 + 36 + i * row_h
        draw.text((x1 + 12, y), label[:28], font=get_font(11), fill=(226, 232, 240))
        w = (bar_x2 - bar_x1) * (abs(val) / max_abs)
        color = (34, 197, 94) if val >= 0 else (239, 68, 68)
        if val >= 0:
            draw.rectangle([bar_x1, y + 4, bar_x1 + w, y + 14], fill=color)
        else:
            draw.rectangle([bar_x1 - w, y + 4, bar_x1, y + 14], fill=color)
        draw.line([bar_x1, y + 2, bar_x1, y + 16], fill=(100, 116, 139))
        draw.text((bar_x2 + 6, y + 2), f"${val:,.2f}", font=get_font(11), fill=(203, 213, 225))


def render_frame(
    step: int,
    total_steps: int,
    config: OptimizerConfig,
    metric_rows: List[Dict[str, Any]],
    batch: pd.DataFrame,
    model: "BidValueModel",
    dq: Dict[str, Any],
    audit: List[Dict[str, Any]],
    drift_score: float,
) -> Image.Image:
    W, H = 1120, 650
    img = Image.new("RGB", (W, H), (2, 6, 23))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 82], fill=(15, 23, 42))
    d.text((24, 14), APP_NAME, font=get_font(28), fill=(226, 232, 240))
    d.text(
        (24, 50),
        f"Scenario {config.scenario} | Policy {config.policy} | Risk {config.risk_mode} | "
        f"Step {step + 1}/{total_steps} | Seed {config.seed} | Model {model.version if model.trained else 'fallback'}",
        font=get_font(14),
        fill=(148, 163, 184),
    )

    rounded_rect(d, [W - 190, 18, W - 24, 56], radius=14, fill=(124, 45, 18), outline=(251, 146, 60), width=2)
    d.text((W - 174, 27), "SYNTHETIC DEMO", font=get_font(16), fill=(255, 255, 255))

    last = metric_rows[-1] if metric_rows else {}
    cum_opt = safe_float(last.get("cum_optimizer_revenue_usd", 0.0))
    cum_base = safe_float(last.get("cum_baseline_revenue_usd", 0.0))
    cum_up = cum_opt - cum_base
    uplift_pct = (cum_up / max(abs(cum_base), 1e-9)) * 100.0
    fill_rate = safe_float(last.get("fill_rate", 0.0))
    avg_floor = safe_float(last.get("avg_floor_cpm", config.base_floor_cpm))
    quality = safe_float(dq.get("quality_score", 0.0))

    cards = [
        ("Cumulative revenue", fmt_usd(cum_opt), "Simulated optimizer outcome", (34, 197, 94)),
        ("Incremental uplift", fmt_usd(cum_up), f"{uplift_pct:.1f}% vs static baseline", (59, 130, 246) if cum_up >= 0 else (239, 68, 68)),
        ("Fill rate", fmt_pct(fill_rate), f"target {fmt_pct(config.min_fill_rate)}", (34, 211, 238)),
        ("Avg floor", f"${avg_floor:.2f} CPM", f"range ${config.min_floor_cpm:.2f}-${config.max_floor_cpm:.2f}", (251, 191, 36)),
        ("Data quality", fmt_pct(quality), f"drift {drift_score:.2f}", (167, 139, 250)),
    ]

    cw, gap, x0, y0 = 205, 12, 24, 98
    for i, c in enumerate(cards):
        draw_card(d, [x0 + i * (cw + gap), y0, x0 + i * (cw + gap) + cw, y0 + 108], *c)

    draw_series_chart(
        d,
        [24, 222, 620, 460],
        metric_rows,
        [
            ("cum_optimizer_revenue_usd", (34, 197, 94), "Optimizer"),
            ("cum_baseline_revenue_usd", (148, 163, 184), "Static baseline"),
        ],
        "Cumulative simulated revenue (USD)",
    )

    rounded_rect(d, [636, 222, 1096, 460], radius=14, fill=(17, 24, 39), outline=(51, 65, 85), width=2)
    d.text((650, 232), "Decision evidence (latest tick)", font=get_font(15), fill=(226, 232, 240))

    y = 260
    if not batch.empty:
        show = batch.tail(5)
        for _, row in show.iterrows():
            seg = str(row.get("segment_key", ""))[:36]
            floor_v = safe_float(row.get("bid_floor_cpm", 0.0))
            pred_v = safe_float(row.get("predicted_max_bid_cpm", 0.0))
            unc_v = safe_float(row.get("prediction_uncertainty_cpm", 0.0))
            reason = str(row.get("decision_reason", ""))[:58]
            approval = str(row.get("approval_status", ""))[:30]
            d.text((650, y), seg, font=get_font(12), fill=(248, 250, 252))
            y += 17
            d.text((650, y), f"floor ${floor_v:.2f} | pred ${pred_v:.2f} ±{unc_v:.2f}", font=get_font(11), fill=(148, 163, 184))
            y += 16
            d.text((650, y), f"{reason} | {approval}", font=get_font(11), fill=(251, 191, 36))
            y += 24

    d.text((650, 404), "Guardrails / audit", font=get_font(13), fill=(226, 232, 240))
    alerts = [a for a in audit if a.get("severity") in ("warn", "error")][-2:]
    if not alerts and audit:
        alerts = audit[-2:]
    yy = 424
    for a in alerts:
        txt = f"{a.get('action', 'info')}: {str(a.get('detail', ''))[:62]}"
        color = (248, 113, 113) if a.get("severity") == "warn" else (148, 163, 184)
        d.text((650, yy), txt, font=get_font(11), fill=color)
        yy += 18

    draw_series_chart(
        d,
        [24, 476, 500, 620],
        metric_rows,
        [
            ("fill_rate", (59, 130, 246), "Fill"),
            ("baseline_fill_rate", (148, 163, 184), "Baseline fill"),
        ],
        "Fill rate",
    )

    top_segments: List[Tuple[str, float]] = []
    if not batch.empty:
        seg = batch.groupby("segment_key")[["revenue_cpm", "baseline_revenue_cpm"]].sum().reset_index()
        seg["uplift_usd"] = (seg["revenue_cpm"] - seg["baseline_revenue_cpm"]) / 1000.0
        seg = seg.sort_values("uplift_usd", key=lambda s: s.abs(), ascending=False)
        top_segments = [(str(r["segment_key"]), safe_float(r["uplift_usd"])) for _, r in seg.head(5).iterrows()]

    draw_hbar(d, [516, 476, 820, 620], top_segments, "Top segment uplift this tick (USD)")

    rounded_rect(d, [836, 476, 1096, 620], radius=14, fill=(17, 24, 39), outline=(51, 65, 85), width=2)
    d.text((850, 486), "Reliability", font=get_font(14), fill=(226, 232, 240))
    stats = [
        f"Data quality: {quality * 100:.0f}%",
        f"Drift score: {drift_score:.2f}",
        f"Latency: {safe_float(last.get('latency_ms', 0.0)):.0f} ms",
        f"Cum fallbacks: {int(safe_float(last.get('cum_fallback_count', 0.0)))}",
        f"Pending approvals: {int(safe_float(last.get('cum_pending_approvals', 0.0)))}",
        f"Guardrail hits: {int(safe_float(last.get('cum_guardrail_violations', 0.0)))}",
    ]
    yy = 512
    for s in stats:
        d.text((850, yy), s, font=get_font(12), fill=(203, 213, 225))
        yy += 18

    d.rectangle([0, H - 28, W, H], fill=(15, 23, 42))
    d.text(
        (24, H - 22),
        "Observed=simulated auction outcome | Estimated=model prediction | Simulated=paired counterfactual | "
        "Predicted=bid value model | No real user data",
        font=get_font(12),
        fill=(148, 163, 184),
    )
    return img


def generate_synthetic_impressions(
    n: int,
    event_time: Any,
    step_idx: int,
    scenario: str,
    rng: np.random.Generator,
    is_history: bool,
) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame(columns=ALL_COLUMNS)

    base_time = pd.Timestamp(event_time)
    countries = rng.choice(COUNTRIES, size=n, p=COUNTRY_PROB)
    devices = rng.choice(DEVICES, size=n, p=DEVICE_PROB)
    segments = rng.choice(SEGMENTS, size=n, p=SEGMENT_PROB)
    demand_sources = rng.choice(DEMAND_SOURCES, size=n, p=DEMAND_PROB)

    if is_history:
        offsets = rng.integers(0, 360, size=n)
        times = pd.Series(base_time + pd.to_timedelta(offsets, unit="m"))
    else:
        times = pd.Series([base_time] * n)

    hour = times.dt.hour.to_numpy(dtype=float)

    country_f = pd.Series(countries).map(COUNTRY_VALUE).to_numpy(dtype=float)
    device_f = pd.Series(devices).map(DEVICE_VALUE).to_numpy(dtype=float)
    segment_f = pd.Series(segments).map(SEGMENT_VALUE).to_numpy(dtype=float)
    demand_f = pd.Series(demand_sources).map(DEMAND_VALUE).to_numpy(dtype=float)

    hour_factor = 1.0 + 0.18 * np.sin(2 * np.pi * (hour - 9) / 24.0) + 0.07 * np.cos(2 * np.pi * hour / 12.0)
    true_cpm = 5.1 * country_f * device_f * segment_f * demand_f * hour_factor

    consent_prob = 0.35 if scenario == "consent_drop" else 0.04
    consent = rng.random(n) > consent_prob
    true_cpm = np.where(consent, true_cpm, true_cpm * 0.91)

    if scenario == "model_drift" and (not is_history) and step_idx > 25:
        drift_mask = (segments == "finance") | (countries == "US")
        true_cpm = np.where(drift_mask, true_cpm * 1.45, true_cpm * 0.92)

    fraud_prob = 0.18 if scenario == "fraud_spike" else 0.012
    fraud = rng.random(n) < fraud_prob

    noise_sigma = 0.26
    if scenario == "high_volatility":
        noise_sigma = 0.55
    if scenario == "low_data":
        noise_sigma = 0.34

    noise = rng.lognormal(0.0, noise_sigma, size=n)
    bid_noise = rng.normal(1.0, 0.07, size=n).clip(0.55, 1.55)
    max_bid = true_cpm * noise * bid_noise

    second_ratio = rng.uniform(0.42, 0.89, size=n)
    second_bid = max_bid * second_ratio

    latency = rng.gamma(2.0, 7.5, size=n) + np.where(fraud, 28.0, 0.0)

    if is_history:
        floor = np.clip(rng.normal(4.1, 1.15, size=n), 0.5, 18.0)
        cleared = max_bid >= floor
        paid = np.where(cleared, np.maximum(floor, second_bid), 0.0)
        paid = np.where(fraud, 0.0, paid)
        decision_reason = "historical_baseline"
        approval_status = "historical"
        model_version = "synthetic-history"
    else:
        floor = np.full(n, np.nan, dtype=float)
        cleared = np.zeros(n, dtype=bool)
        paid = np.zeros(n, dtype=float)
        decision_reason = ""
        approval_status = ""
        model_version = ""

    demand_out = np.array(demand_sources, dtype=object)
    if scenario == "schema_drift" and (not is_history) and step_idx % 8 == 0:
        mask = rng.random(n) < 0.45
        demand_out = demand_out.copy()
        demand_out[mask] = np.nan

    segment_keys = [
        segment_key_from_row(c, d, s, dem, ct)
        for c, d, s, dem, ct in zip(countries, devices, segments, demand_out, consent)
    ]

    df = pd.DataFrame(
        {
            "impression_uid": [str(uuid.uuid4()) for _ in range(n)],
            "event_time": times,
            "publisher_id": rng.choice(PUBLISHERS, size=n),
            "ad_unit": rng.choice(AD_UNITS, size=n),
            "country": countries,
            "device": devices,
            "audience_segment": segments,
            "demand_source": demand_out,
            "consent_state": consent.astype(bool),
            "fraud_flag": fraud.astype(bool),
            "bid_floor_cpm": floor,
            "max_bid_cpm": max_bid,
            "second_bid_cpm": second_bid,
            "winning_bid_cpm": paid,
            "revenue_cpm": paid,
            "cleared": cleared.astype(bool),
            "latency_ms": latency,
            "segment_key": segment_keys,
            "model_version": model_version,
            "experiment_arm": rng.choice(EXPERIMENT_ARMS, size=n),
            "decision_reason": decision_reason,
            "approval_status": approval_status,
        }
    )

    for col in AUX_COLUMNS:
        df[col] = np.nan

    return df


def validate_and_clean(df: pd.DataFrame, watermark: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "input_rows": int(len(df)),
        "duplicate_uid": 0,
        "malformed_timestamps": 0,
        "negative_values": 0,
        "late_arrivals": 0,
        "outliers_clipped": 0,
        "schema_added": [],
        "missing_cell_rate": 0.0,
        "quality_score": 1.0,
        "rows_retained": int(len(df)),
        "issues": [],
    }

    df = df.copy()

    if df.empty:
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)
        report["quality_score"] = 0.0
        report["issues"].append("empty_input")
        return df, report

    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
            report["schema_added"].append(c)
            report["issues"].append(f"schema_added:{c}")

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    bad_time = df["event_time"].isna()
    if bad_time.any():
        fill_time = watermark if watermark is not None else pd.Timestamp.now(tz="UTC")
        df.loc[bad_time, "event_time"] = fill_time
        report["malformed_timestamps"] += int(bad_time.sum())
        report["issues"].append("malformed_timestamps")

    numeric_cols = [
        "bid_floor_cpm",
        "max_bid_cpm",
        "second_bid_cpm",
        "winning_bid_cpm",
        "revenue_cpm",
        "latency_ms",
        "predicted_max_bid_cpm",
        "prediction_uncertainty_cpm",
        "baseline_revenue_cpm",
    ]
    for c in numeric_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["revenue_cpm", "winning_bid_cpm", "latency_ms", "max_bid_cpm", "second_bid_cpm", "baseline_revenue_cpm"]:
        if c in df:
            neg = df[c] < 0
            if neg.any():
                report["negative_values"] += int(neg.sum())
                df.loc[neg, c] = np.abs(df.loc[neg, c])
                report["issues"].append(f"negative_values:{c}")

    if "bid_floor_cpm" in df:
        bad_floor = df["bid_floor_cpm"] < 0
        df.loc[bad_floor, "bid_floor_cpm"] = np.nan

    for c in ["consent_state", "fraud_flag", "cleared"]:
        if c in df:
            df[c] = df[c].map(lambda x: bool(x) if pd.notna(x) else False).astype(bool)

    string_cols = [
        "impression_uid",
        "publisher_id",
        "ad_unit",
        "country",
        "device",
        "audience_segment",
        "demand_source",
        "segment_key",
        "model_version",
        "experiment_arm",
        "decision_reason",
        "approval_status",
    ]
    for c in string_cols:
        if c in df:
            df[c] = df[c].fillna("").astype(str)

    dup = df.duplicated(subset=["impression_uid"], keep="first")
    report["duplicate_uid"] = int(dup.sum())
    if report["duplicate_uid"]:
        report["issues"].append("duplicate_uid")
    df = df.loc[~dup].copy()

    if watermark is not None and not df.empty:
        late = df["event_time"] < (watermark - pd.Timedelta(minutes=2))
        report["late_arrivals"] = int(late.sum())
        if report["late_arrivals"]:
            report["issues"].append("late_arrivals")

    if not df.empty:
        out = (df["revenue_cpm"] > 250) | (df["latency_ms"] > 5000) | (df["max_bid_cpm"] > 500)
        report["outliers_clipped"] = int(out.sum())
        if report["outliers_clipped"]:
            report["issues"].append("outliers_clipped")
        df.loc[out, "revenue_cpm"] = df.loc[out, "revenue_cpm"].clip(upper=250)
        df.loc[out, "latency_ms"] = df.loc[out, "latency_ms"].clip(upper=5000)
        df.loc[out, "max_bid_cpm"] = df.loc[out, "max_bid_cpm"].clip(upper=500)

        if df["segment_key"].eq("").any():
            df["segment_key"] = df.apply(
                lambda r: segment_key_from_row(r["country"], r["device"], r["audience_segment"], r["demand_source"], bool(r["consent_state"])),
                axis=1,
            )

    report["rows_retained"] = int(len(df))
    missing_rate = float(df.isna().mean().mean()) if not df.empty else 1.0
    report["missing_cell_rate"] = missing_rate

    issue_penalty = (
        report["duplicate_uid"] * 0.01
        + report["malformed_timestamps"] * 0.02
        + report["negative_values"] * 0.01
        + report["late_arrivals"] * 0.005
        + report["outliers_clipped"] * 0.02
        + len(report["schema_added"]) * 0.08
    )
    denom = max(report["input_rows"], 1)
    report["quality_score"] = float(np.clip(1.0 - issue_penalty / denom - missing_rate * 0.35, 0.0, 1.0))

    return df, report


def encode_series(s: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    return s.fillna("unknown").astype(str).map(mapping).fillna(0).astype(float)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["country_enc"] = encode_series(df["country"], COUNTRY_ENC)
    X["device_enc"] = encode_series(df["device"], DEVICE_ENC)

    seg = np.where(df["consent_state"].astype(bool), df["audience_segment"].fillna("unknown"), "unknown")
    X["segment_enc"] = pd.Series(seg, index=df.index).astype(str).map(SEGMENT_ENC).fillna(0).astype(float)

    X["demand_enc"] = encode_series(df["demand_source"], DEMAND_ENC)

    hour = df["event_time"].dt.hour.fillna(0).astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    X["consent_int"] = df["consent_state"].astype(float)
    X["fraud_int"] = df["fraud_flag"].astype(float)
    X["latency_norm"] = df["latency_ms"].fillna(0).clip(0, 5000) / 1000.0

    return X


class BidValueModel:
    def __init__(self, seeds: Tuple[int, ...] = (11, 23, 37), max_iter: int = 90):
        self.seeds = seeds
        self.max_iter = max_iter
        self.models: List[HistGradientBoostingRegressor] = []
        self.feature_names: List[str] = []
        self.trained = False
        self.version = MODEL_FAMILY
        self.residual_std = 1.0
        self.val_metrics: Dict[str, Any] = {}
        self.val_data: Optional[pd.DataFrame] = None
        self.val_pred: Optional[np.ndarray] = None
        self.val_unc: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> None:
        self.val_data = None
        self.val_pred = None
        self.val_unc = None

        train_all = df[(~df["fraud_flag"].astype(bool)) & (df["max_bid_cpm"].notna()) & (df["max_bid_cpm"] >= 0)].copy()
        if len(train_all) < 120:
            self.trained = False
            self.val_metrics = {"status": "insufficient_data", "rows": int(len(train_all))}
            return

        train_all = train_all.sort_values("event_time")
        split = int(len(train_all) * 0.8)
        tr = train_all.iloc[:split]
        va = train_all.iloc[split:]

        Xtr = featurize(tr)
        ytr = tr["max_bid_cpm"].to_numpy(dtype=float)

        self.models = []
        for seed in self.seeds:
            m = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=0.07,
                min_samples_leaf=12,
                l2_regularization=0.1,
                random_state=seed,
            )
            m.fit(Xtr, ytr)
            self.models.append(m)

        self.feature_names = list(Xtr.columns)

        if len(va) >= 20:
            Xva = featurize(va).reindex(columns=self.feature_names, fill_value=0)
            preds = np.column_stack([m.predict(Xva) for m in self.models])
            pred = preds.mean(axis=1)
            yva = va["max_bid_cpm"].to_numpy(dtype=float)

            mae = float(mean_absolute_error(yva, pred))
            rmse = float(np.sqrt(mean_squared_error(yva, pred)))
            try:
                r2 = float(r2_score(yva, pred))
            except Exception:
                r2 = None

            ens_var = preds.var(axis=1)
            residual = float(np.std(yva - pred))
            unc = np.sqrt(ens_var + (residual * 0.35) ** 2)
            coverage = float(np.mean(np.abs(yva - pred) <= 1.96 * unc))

            self.residual_std = residual
            self.val_metrics = {
                "status": "trained",
                "rows_train": int(len(tr)),
                "rows_validation": int(len(va)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "coverage_95": coverage,
            }
            self.val_data = va[
                ["event_time", "country", "device", "audience_segment", "demand_source", "consent_state", "max_bid_cpm"]
            ].copy()
            self.val_pred = pred
            self.val_unc = unc
        else:
            train_preds = np.column_stack([m.predict(Xtr) for m in self.models]).mean(axis=1)
            self.residual_std = float(np.std(ytr - train_preds)) if len(ytr) > 1 else 1.0
            self.val_metrics = {"status": "small_validation", "rows_train": int(len(tr)), "rows_validation": int(len(va))}

        self.trained = True
        mae_for_version = self.val_metrics.get("mae", 0.0)
        self.version = f"{MODEL_FAMILY}-{safe_float(mae_for_version):.3f}"

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(df)
        if not self.trained or n == 0:
            return np.full(n, np.nan), np.full(n, np.nan)

        X = featurize(df).reindex(columns=self.feature_names, fill_value=0)
        preds = np.column_stack([m.predict(X) for m in self.models])
        mean = preds.mean(axis=1)
        ens_std = preds.std(axis=1)
        unc = np.sqrt(ens_std ** 2 + (self.residual_std * 0.35) ** 2)
        unc = np.where(unc < 0.15, 0.15, unc)
        return mean, unc


def default_state() -> Dict[str, Any]:
    return {
        "prev_floor": None,
        "max_history": deque(maxlen=250),
        "second_history": deque(maxlen=250),
        "fill_ema": 0.65,
        "rev_ema": 4.0,
        "approved_count": 0,
        "fallback_count": 0,
        "rate_limited_count": 0,
        "exploration_count": 0,
        "pending_count": 0,
    }


def build_candidate_floors(base: float, mean: float, std: float, hist_max: np.ndarray, config: OptimizerConfig) -> List[float]:
    candidates = {
        base,
        config.min_floor_cpm + 0.25,
        config.max_floor_cpm * 0.25,
        base * 1.2,
    }

    if not math.isnan(mean):
        for mult in [0.55, 0.70, 0.85, 1.00, 1.15]:
            candidates.add(mean * mult)
        if not math.isnan(std):
            candidates.add(mean + 0.25 * std)
            candidates.add(mean - 0.35 * std)

    if len(hist_max) >= 10:
        for q in [0.50, 0.65, 0.78, 0.90]:
            candidates.add(float(np.quantile(hist_max, q)))

    out = []
    for c in candidates:
        c = safe_float(c, np.nan)
        if not math.isnan(c) and config.min_floor_cpm <= c <= config.max_floor_cpm:
            out.append(round(float(c), 3))

    if not out:
        out = [round(float(config.base_floor_cpm), 3)]

    return sorted(set(out))


def choose_floor(
    key: str,
    state: Dict[str, Any],
    mean: Any,
    std: Any,
    config: OptimizerConfig,
    rng: np.random.Generator,
    dq_score: float,
    policy: str,
    step_idx: int,
) -> Dict[str, Any]:
    prev = safe_float(state.get("prev_floor"), config.base_floor_cpm)
    mean_val = finite_or_nan(mean)
    std_val = finite_or_nan(std)

    hist_max = np.array(list(state.get("max_history", [])), dtype=float)
    hist_second = np.array(list(state.get("second_history", [])), dtype=float)
    hist_max = hist_max[np.isfinite(hist_max)]
    hist_second = hist_second[np.isfinite(hist_second)]

    reasons: List[str] = []
    approval = "not_required"
    fallback = False
    rate_limited = False
    exploration = False
    pending = False
    approved = False

    low_quality = dq_score < 0.62 or len(hist_max) < 12
    high_unc = (
        not math.isnan(std_val)
        and std_val > max(2.2, 0.55 * max(mean_val if not math.isnan(mean_val) else config.base_floor_cpm, 1.0))
    )
    conservative = config.risk_mode == "conservative"

    if policy == "static":
        target = config.base_floor_cpm
        reasons.append("static_baseline")
    elif low_quality or (conservative and high_unc):
        if len(hist_max) >= 5:
            target = float(np.median(hist_max) * 0.72)
        else:
            target = config.base_floor_cpm
        reasons.append("deterministic_fallback_low_quality_or_high_uncertainty")
        state["fallback_count"] = int(state.get("fallback_count", 0)) + 1
        fallback = True
    else:
        candidates = build_candidate_floors(config.base_floor_cpm, mean_val, std_val, hist_max, config)

        if policy == "quantile":
            if len(hist_max) >= 10:
                target = float(np.quantile(hist_max, 0.78))
            else:
                target = config.base_floor_cpm
            reasons.append("rolling_quantile_rule")
        else:
            sample_mean = mean_val if not math.isnan(mean_val) else config.base_floor_cpm * 1.25
            sample_std = std_val if not math.isnan(std_val) else 1.0

            scored = []
            for f in candidates:
                if len(hist_max) >= 30 and len(hist_second) >= 30:
                    hm = hist_max[-200:]
                    hs = hist_second[-200:]
                    min_len = min(len(hm), len(hs))
                    hm = hm[:min_len]
                    hs = hs[:min_len]
                    cleared = hm >= f
                    paid = np.where(cleared, np.maximum(f, hs), 0.0)
                    fill = float(cleared.mean())
                    rev = float(paid.mean())
                else:
                    n_sample = 160
                    max_samples = rng.normal(sample_mean, max(sample_std, 0.15), n_sample).clip(0.05, None)
                    second_samples = max_samples * rng.uniform(0.42, 0.89, n_sample)
                    cleared = max_samples >= f
                    paid = np.where(cleared, np.maximum(f, second_samples), 0.0)
                    fill = float(cleared.mean())
                    rev = float(paid.mean())

                penalty = max(0.0, config.min_fill_rate - fill) * 12.0
                unc_pen = 0.0
                if policy == "autonomous" and not math.isnan(std_val):
                    weight = 0.02 if config.risk_mode == "balanced" else 0.06 if config.risk_mode == "conservative" else 0.005
                    unc_pen = weight * std_val * (f / max(prev, 1.0))

                scored.append((rev - penalty - unc_pen, f, fill, rev))

            scored.sort(key=lambda x: x[0], reverse=True)
            target = float(scored[0][1])
            reasons.append("expected_revenue_optimizer")

            if policy == "autonomous" and rng.random() < config.exploration_rate:
                alt = [c for c in candidates if abs(c - target) > 0.15]
                if alt:
                    target = float(rng.choice(alt))
                    reasons.append("bounded_exploration")
                    state["exploration_count"] = int(state.get("exploration_count", 0)) + 1
                    exploration = True

    target = float(np.clip(target, config.min_floor_cpm, config.max_floor_cpm))
    attempt_target = target

    max_step = max(prev * config.max_change_pct / 100.0, 0.08)
    if abs(target - prev) > max_step:
        target = prev + math.copysign(max_step, target - prev)
        target = float(np.clip(target, config.min_floor_cpm, config.max_floor_cpm))
        reasons.append("rate_limited")
        state["rate_limited_count"] = int(state.get("rate_limited_count", 0)) + 1
        rate_limited = True

    delta_pct = 0.0 if prev <= 0 else abs(attempt_target - prev) / max(prev, 0.01) * 100.0

    if policy == "autonomous" and delta_pct > config.require_approval_above_pct and config.risk_mode != "aggressive":
        if config.simulate_human_approval:
            approval = "simulated_human_approved"
            state["approved_count"] = int(state.get("approved_count", 0)) + 1
            approved = True
        else:
            approval = "pending_human_approval"
            target = prev
            reasons.append("held_for_human_approval")
            state["pending_count"] = int(state.get("pending_count", 0)) + 1
            pending = True
    elif delta_pct > 1.0:
        approval = "auto_within_guardrail"

    return {
        "floor": float(target),
        "reason": "; ".join(reasons),
        "approval": approval,
        "delta_pct": float(delta_pct),
        "previous_floor": float(prev),
        "fallback": fallback,
        "rate_limited": rate_limited,
        "exploration": exploration,
        "pending": pending,
        "approved": approved,
    }


def encode_frames(frames: List[np.ndarray], fps: int, workdir: str) -> Tuple[Optional[str], Optional[str]]:
    if not frames:
        return None, None

    mp4_path = os.path.join(workdir, "autonomous_ad_optimizer_demo.mp4")
    gif_path = os.path.join(workdir, "autonomous_ad_optimizer_demo.gif")

    if av is not None:
        try:
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            container = av.open(mp4_path, mode="w", format="mp4")
            stream = container.add_stream("libx264", rate=max(int(fps), 1))
            stream.width = int(frames[0].shape[1])
            stream.height = int(frames[0].shape[0])
            stream.pix_fmt = "yuv420p"
            try:
                stream.options = {"crf": "30", "preset": "ultrafast"}
            except Exception:
                pass

            for arr in frames:
                arr = np.ascontiguousarray(arr, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
            container.close()

            if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000:
                return mp4_path, None
        except Exception:
            try:
                container.close()
            except Exception:
                pass

    try:
        imgs = [Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).convert("P", palette=Image.ADAPTIVE) for arr in frames]
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / max(int(fps), 1)),
            loop=0,
            optimize=True,
        )
        if os.path.exists(gif_path) and os.path.getsize(gif_path) > 1000:
            return None, gif_path
    except Exception:
        pass

    return None, None


def run_simulation(
    config: OptimizerConfig,
    workdir: str,
    fast: bool = False,
    render_video: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(config.seed))
    now = pd.Timestamp.now(tz="UTC")

    history_size = int(config.history_size)
    if config.scenario == "low_data":
        history_size = max(180, int(history_size * 0.25))

    hist_start = now - pd.Timedelta(hours=6)
    history = generate_synthetic_impressions(history_size, hist_start, 0, config.scenario, rng, True)
    history, hist_dq = validate_and_clean(history, watermark=now)

    if fast:
        model = BidValueModel(seeds=(11,), max_iter=40)
    else:
        model = BidValueModel(seeds=(11, 23, 37), max_iter=90)
    model.fit(history)

    states: Dict[str, Dict[str, Any]] = {}
    history_sorted = history.sort_values("event_time")

    for _, row in history_sorted.iterrows():
        key = str(row["segment_key"])
        state = states.setdefault(key, default_state())
        mx = finite_or_nan(row["max_bid_cpm"])
        sec = finite_or_nan(row["second_bid_cpm"])
        if not bool(row["fraud_flag"]) and not math.isnan(mx):
            state["max_history"].append(mx)
            state["second_history"].append(sec if not math.isnan(sec) else mx * 0.7)
        fill = 1.0 if bool(row["cleared"]) else 0.0
        state["fill_ema"] = 0.9 * state["fill_ema"] + 0.1 * fill
        state["rev_ema"] = 0.9 * state["rev_ema"] + 0.1 * safe_float(row["revenue_cpm"])
        pf = finite_or_nan(row["bid_floor_cpm"])
        if not math.isnan(pf):
            state["prev_floor"] = pf

    steps = max(4, int(config.duration_sec * config.fps))
    if fast:
        steps = max(3, int(config.duration_sec * config.fps))

    frames: List[np.ndarray] = []
    metric_rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    all_batches: List[pd.DataFrame] = []
    pred_errors = deque(maxlen=60)

    cum_rev = 0.0
    cum_base = 0.0
    cum_imp = 0
    cum_cleared = 0
    cum_base_cleared = 0
    cum_latency = 0.0
    dq_sum = 0.0

    cum_fallback = 0
    cum_pending = 0
    cum_approved = 0
    cum_rate_limited = 0
    cum_exploration = 0
    cum_violations = 0

    last_dq = hist_dq

    for step in range(steps):
        event_time = now + pd.Timedelta(seconds=step / max(float(config.fps), 1.0))
        batch_n = int(config.batch_size)
        if config.scenario == "low_data":
            batch_n = max(8, int(batch_n * 0.30))

        batch = generate_synthetic_impressions(batch_n, event_time, step, config.scenario, rng, False)
        batch, dq = validate_and_clean(batch, watermark=event_time)
        last_dq = dq

        if batch.empty:
            continue

        mean, std = model.predict(batch)
        batch["predicted_max_bid_cpm"] = mean
        batch["prediction_uncertainty_cpm"] = std

        recent_drift = float(np.mean(list(pred_errors)[-20:])) if len(pred_errors) >= 20 else 0.0
        effective_dq = float(dq["quality_score"])
        drift_threshold = max(1.8, 1.35 * safe_float(getattr(model, "residual_std", 1.0), 1.0))

        if recent_drift > drift_threshold:
            effective_dq = min(effective_dq, 0.58)
            if step % 7 == 0:
                audit.append(
                    {
                        "timestamp": event_time.isoformat(),
                        "actor": "monitor",
                        "action": "model_drift_detected",
                        "severity": "warn",
                        "detail": f"recent MAE {recent_drift:.2f} > threshold {drift_threshold:.2f}; conservative fallback engaged",
                        "model_version": model.version,
                        "data_quality": effective_dq,
                    }
                )

        decisions: List[Dict[str, Any]] = []
        for _, row in batch.iterrows():
            key = str(row["segment_key"])
            state = states.setdefault(key, default_state())
            decision = choose_floor(
                key,
                state,
                row["predicted_max_bid_cpm"],
                row["prediction_uncertainty_cpm"],
                config,
                rng,
                effective_dq,
                config.policy,
                step,
            )
            decisions.append(decision)

            severity = "warn" if (
                decision["fallback"]
                or decision["pending"]
                or decision["rate_limited"]
                or "drift" in decision["reason"]
            ) else "info"

            if decision["delta_pct"] > 1.0 or severity == "warn" or len(audit) < 5:
                audit.append(
                    {
                        "timestamp": event_time.isoformat(),
                        "actor": "optimizer_agent",
                        "action": "floor_decision",
                        "severity": severity,
                        "segment": key,
                        "previous_floor": decision["previous_floor"],
                        "new_floor": decision["floor"],
                        "delta_pct": decision["delta_pct"],
                        "reason": decision["reason"],
                        "approval_status": decision["approval"],
                        "model_version": model.version,
                        "data_quality": effective_dq,
                        "detail": f"{key}: {decision['previous_floor']:.2f}->{decision['floor']:.2f} ({decision['delta_pct']:.1f}%) {decision['reason']}",
                    }
                )

        batch["bid_floor_cpm"] = [d["floor"] for d in decisions]
        batch["decision_reason"] = [d["reason"] for d in decisions]
        batch["approval_status"] = [d["approval"] for d in decisions]
        batch["_fallback"] = [d["fallback"] for d in decisions]
        batch["_rate_limited"] = [d["rate_limited"] for d in decisions]
        batch["_exploration"] = [d["exploration"] for d in decisions]
        batch["_pending"] = [d["pending"] for d in decisions]
        batch["_approved"] = [d["approved"] for d in decisions]

        cleared = batch["max_bid_cpm"] >= batch["bid_floor_cpm"]
        paid = np.where(cleared, np.maximum(batch["bid_floor_cpm"], batch["second_bid_cpm"]), 0.0)
        paid = np.where(batch["fraud_flag"], 0.0, paid)

        batch["cleared"] = cleared.astype(bool)
        batch["winning_bid_cpm"] = paid
        batch["revenue_cpm"] = paid

        base_floor = float(config.base_floor_cpm)
        base_cleared = batch["max_bid_cpm"] >= base_floor
        base_paid = np.where(base_cleared, np.maximum(base_floor, batch["second_bid_cpm"]), 0.0)
        base_paid = np.where(batch["fraud_flag"], 0.0, base_paid)
        batch["baseline_revenue_cpm"] = base_paid

        nonfraud = ~batch["fraud_flag"].astype(bool)
        if model.trained and nonfraud.any():
            actual = batch.loc[nonfraud, "max_bid_cpm"].to_numpy(dtype=float)
            pred = batch.loc[nonfraud, "predicted_max_bid_cpm"].to_numpy(dtype=float)
            errs = np.abs(actual - pred)
            for e in errs:
                ef = safe_float(e, np.nan)
                if not math.isnan(ef):
                    pred_errors.append(float(ef))

        for idx, row in batch.iterrows():
            state = states[str(row["segment_key"])]
            mx = finite_or_nan(row["max_bid_cpm"])
            sec = finite_or_nan(row["second_bid_cpm"])
            if not bool(row["fraud_flag"]) and not math.isnan(mx):
                state["max_history"].append(mx)
                state["second_history"].append(sec if not math.isnan(sec) else mx * 0.7)
            fill = 1.0 if bool(row["cleared"]) else 0.0
            state["fill_ema"] = 0.85 * state["fill_ema"] + 0.15 * fill
            state["rev_ema"] = 0.85 * state["rev_ema"] + 0.15 * safe_float(row["revenue_cpm"])
            state["prev_floor"] = safe_float(row["bid_floor_cpm"], state.get("prev_floor") or config.base_floor_cpm)

        rev = float(batch["revenue_cpm"].sum() / 1000.0)
        base_rev = float(batch["baseline_revenue_cpm"].sum() / 1000.0)
        imp = int(len(batch))
        cleared_count = int(batch["cleared"].sum())
        base_cleared_count = int(base_cleared.sum())

        fill = float(batch.loc[nonfraud, "cleared"].mean()) if nonfraud.any() else 0.0
        base_fill = float(base_cleared[nonfraud].mean()) if nonfraud.any() else 0.0
        rpm = float(batch["revenue_cpm"].mean())
        base_rpm = float(batch["baseline_revenue_cpm"].mean())
        avg_floor = float(batch["bid_floor_cpm"].mean())
        avg_latency = float(batch["latency_ms"].mean())

        tick_fallback = int(batch["_fallback"].sum())
        tick_pending = int(batch["_pending"].sum())
        tick_approved = int(batch["_approved"].sum())
        tick_rate = int(batch["_rate_limited"].sum())
        tick_explore = int(batch["_exploration"].sum())

        tick_violations = 0
        if fill < config.min_fill_rate:
            tick_violations += 1
        if avg_floor < config.min_floor_cpm - 1e-6 or avg_floor > config.max_floor_cpm + 1e-6:
            tick_violations += 1

        cum_rev += rev
        cum_base += base_rev
        cum_imp += imp
        cum_cleared += cleared_count
        cum_base_cleared += base_cleared_count
        cum_latency += avg_latency * imp
        dq_sum += effective_dq

        cum_fallback += tick_fallback
        cum_pending += tick_pending
        cum_approved += tick_approved
        cum_rate_limited += tick_rate
        cum_exploration += tick_explore
        cum_violations += tick_violations

        metric_rows.append(
            {
                "step": step,
                "event_time": event_time,
                "impressions": imp,
                "optimizer_revenue_usd": rev,
                "baseline_revenue_usd": base_rev,
                "uplift_usd": rev - base_rev,
                "cum_optimizer_revenue_usd": cum_rev,
                "cum_baseline_revenue_usd": cum_base,
                "cum_uplift_usd": cum_rev - cum_base,
                "fill_rate": fill,
                "baseline_fill_rate": base_fill,
                "rpm": rpm,
                "baseline_rpm": base_rpm,
                "avg_floor_cpm": avg_floor,
                "latency_ms": avg_latency,
                "data_quality": effective_dq,
                "fraud_rate": float(batch["fraud_flag"].mean()),
                "consent_rate": float(batch["consent_state"].mean()),
                "fallback_count": tick_fallback,
                "cum_fallback_count": cum_fallback,
                "pending_approvals": tick_pending,
                "cum_pending_approvals": cum_pending,
                "simulated_approvals": tick_approved,
                "cum_simulated_approvals": cum_approved,
                "rate_limited_count": tick_rate,
                "cum_rate_limited_count": cum_rate_limited,
                "exploration_count": tick_explore,
                "cum_exploration_count": cum_exploration,
                "guardrail_violations": tick_violations,
                "cum_guardrail_violations": cum_violations,
                "drift_score": recent_drift,
            }
        )

        if render_video:
            frame = render_frame(step, steps, config, metric_rows, batch, model, dq, audit, recent_drift)
            frames.append(np.ascontiguousarray(np.array(frame.convert("RGB")), dtype=np.uint8))

        all_batches.append(batch)

        if (not fast) and render_video and step > 0 and step % 20 == 0 and len(all_batches) > 1:
            pool = pd.concat([history] + all_batches[:-1], ignore_index=True)
            model.fit(pool)
            audit.append(
                {
                    "timestamp": event_time.isoformat(),
                    "actor": "system",
                    "action": "model_refit",
                    "severity": "info",
                    "detail": f"refit on {len(pool)} prior rows; model_version={model.version}",
                    "model_version": model.version,
                    "data_quality": effective_dq,
                }
            )

        if len(audit) > 500:
            audit = audit[-500:]

    impressions_df = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame(columns=ALL_COLUMNS + TEMP_COLUMNS)
    metrics_df = pd.DataFrame(metric_rows)

    mp4_path, gif_path = (None, None)
    final_frame_png = None

    if render_video:
        mp4_path, gif_path = encode_frames(frames, config.fps, workdir)
        if frames:
            try:
                buf = io.BytesIO()
                Image.fromarray(frames[-1]).save(buf, format="PNG")
                final_frame_png = buf.getvalue()
            except Exception:
                final_frame_png = None

    segment_stats = pd.DataFrame()
    if not impressions_df.empty:
        segment_stats = (
            impressions_df.groupby("segment_key")
            .agg(
                impressions=("revenue_cpm", "size"),
                optimizer_revenue_cpm_sum=("revenue_cpm", "sum"),
                baseline_revenue_cpm_sum=("baseline_revenue_cpm", "sum"),
                fill_rate=("cleared", "mean"),
                avg_floor_cpm=("bid_floor_cpm", "mean"),
                avg_predicted_bid=("predicted_max_bid_cpm", "mean"),
                avg_uncertainty=("prediction_uncertainty_cpm", "mean"),
                fallbacks=("_fallback", "sum"),
                pending=("_pending", "sum"),
                approved=("_approved", "sum"),
                rate_limited=("_rate_limited", "sum"),
                exploration=("_exploration", "sum"),
            )
            .reset_index()
        )
        segment_stats["optimizer_revenue_usd"] = segment_stats["optimizer_revenue_cpm_sum"] / 1000.0
        segment_stats["baseline_revenue_usd"] = segment_stats["baseline_revenue_cpm_sum"] / 1000.0
        segment_stats["uplift_usd"] = segment_stats["optimizer_revenue_usd"] - segment_stats["baseline_revenue_usd"]
        segment_stats["uplift_pct"] = segment_stats["uplift_usd"] / segment_stats["baseline_revenue_usd"].replace(0, np.nan) * 100.0
        segment_stats = segment_stats.sort_values("uplift_usd", ascending=False)

    cum_base_safe = max(cum_base, 1e-9)
    summary = {
        "cum_revenue": cum_rev,
        "cum_baseline": cum_base,
        "uplift_usd": cum_rev - cum_base,
        "uplift_pct": (cum_rev - cum_base) / cum_base_safe * 100.0,
        "fill_rate": cum_cleared / max(cum_imp, 1),
        "baseline_fill_rate": cum_base_cleared / max(cum_imp, 1),
        "avg_floor_cpm": float(impressions_df["bid_floor_cpm"].mean()) if not impressions_df.empty else config.base_floor_cpm,
        "latency_ms": cum_latency / max(cum_imp, 1),
        "data_quality": dq_sum / max(steps, 1),
        "pending_approvals": cum_pending,
        "simulated_approvals": cum_approved,
        "fallback_count": cum_fallback,
        "rate_limited_count": cum_rate_limited,
        "exploration_count": cum_exploration,
        "guardrail_violations": cum_violations,
        "model_trained": bool(model.trained),
        "model_version": model.version,
        "steps": steps,
        "frames": len(frames),
        "scenario": config.scenario,
        "policy": config.policy,
        "risk_mode": config.risk_mode,
        "segment_count": int(segment_stats["segment_key"].nunique()) if not segment_stats.empty else 0,
    }

    return {
        "config": as_dict(config),
        "history": history,
        "impressions": impressions_df,
        "metrics": metrics_df,
        "segment_stats": segment_stats,
        "audit": audit,
        "model": model,
        "hist_dq": hist_dq,
        "last_dq": last_dq,
        "mp4_path": mp4_path,
        "gif_path": gif_path,
        "final_frame_png": final_frame_png,
        "summary": summary,
    }


def compare_policies(base_config: OptimizerConfig, workdir: str) -> pd.DataFrame:
    rows = []
    for p in POLICIES:
        cfg = replace_config(
            base_config,
            policy=p,
            duration_sec=3,
            fps=6,
            batch_size=40,
            history_size=1200,
            simulate_human_approval=True,
        )
        res = run_simulation(cfg, workdir, fast=True, render_video=False)
        s = res["summary"]
        rows.append(
            {
                "policy": p,
                "simulated_revenue_usd": s["cum_revenue"],
                "baseline_revenue_usd": s["cum_baseline"],
                "uplift_usd": s["uplift_usd"],
                "uplift_pct": s["uplift_pct"],
                "fill_rate": s["fill_rate"],
                "avg_floor_cpm": s["avg_floor_cpm"],
                "latency_ms": s["latency_ms"],
                "fallbacks": s["fallback_count"],
                "pending_approvals": s["pending_approvals"],
                "simulated_approvals": s["simulated_approvals"],
                "rate_limited": s["rate_limited_count"],
                "exploration": s["exploration_count"],
                "guardrail_violations": s["guardrail_violations"],
                "data_quality": s["data_quality"],
            }
        )
    return pd.DataFrame(rows)


def run_self_tests() -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            checks.append({"test": name, "status": "pass", "error": ""})
        except Exception as e:
            checks.append({"test": name, "status": "fail", "error": str(e)})

    def t_duplicate_detection():
        ts = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame(
            [
                {
                    "impression_uid": "dup",
                    "event_time": ts,
                    "country": "US",
                    "device": "mobile",
                    "audience_segment": "news",
                    "demand_source": "direct",
                    "consent_state": True,
                    "fraud_flag": False,
                    "bid_floor_cpm": 4.0,
                    "max_bid_cpm": 6.0,
                    "second_bid_cpm": 4.5,
                    "winning_bid_cpm": 4.5,
                    "revenue_cpm": 4.5,
                    "cleared": True,
                    "latency_ms": 20.0,
                },
                {
                    "impression_uid": "dup",
                    "event_time": ts,
                    "country": "US",
                    "device": "mobile",
                    "audience_segment": "news",
                    "demand_source": "direct",
                    "consent_state": True,
                    "fraud_flag": False,
                    "bid_floor_cpm": 4.0,
                    "max_bid_cpm": 6.0,
                    "second_bid_cpm": 4.5,
                    "winning_bid_cpm": 4.5,
                    "revenue_cpm": 4.5,
                    "cleared": True,
                    "latency_ms": 20.0,
                },
            ]
        )
        clean, rep = validate_and_clean(df)
        assert rep["duplicate_uid"] == 1
        assert len(clean) == 1

    def t_missing_schema():
        df = pd.DataFrame([{"impression_uid": "a", "event_time": pd.Timestamp.now(tz="UTC")}])
        clean, rep = validate_and_clean(df)
        assert "demand_source" in rep["schema_added"] or "demand_source" in clean.columns

    def t_rate_limit():
        cfg = OptimizerConfig(policy="autonomous", risk_mode="balanced", max_change_pct=10.0, simulate_human_approval=True)
        state = default_state()
        state["prev_floor"] = 5.0
        for _ in range(40):
            state["max_history"].append(20.0)
            state["second_history"].append(15.0)
        out = choose_floor("k", state, 20.0, 1.0, cfg, np.random.default_rng(1), 0.95, "autonomous", 0)
        assert abs(out["floor"] - 5.0) <= 5.0 * 0.10 + 0.12

    def t_fallback_low_quality():
        cfg = OptimizerConfig()
        state = default_state()
        state["prev_floor"] = 4.0
        out = choose_floor("k", state, np.nan, np.nan, cfg, np.random.default_rng(2), 0.20, "autonomous", 0)
        assert "fallback" in out["reason"]
        assert out["fallback"] is True

    def t_synthetic_unique_and_fields():
        rng = np.random.default_rng(3)
        df = generate_synthetic_impressions(20, pd.Timestamp.now(tz="UTC"), 0, "normal", rng, False)
        assert df["impression_uid"].nunique() == 20
        assert "max_bid_cpm" in df
        assert "segment_key" in df

    def t_model_smoke():
        rng = np.random.default_rng(4)
        df = generate_synthetic_impressions(250, pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1), 0, "normal", rng, True)
        df, _ = validate_and_clean(df)
        m = BidValueModel(seeds=(1,), max_iter=20)
        m.fit(df)
        assert m.trained
        pred, unc = m.predict(df.head(10))
        assert len(pred) == 10
        assert len(unc) == 10

    def t_simulation_smoke():
        cfg = OptimizerConfig(duration_sec=1, fps=4, batch_size=20, history_size=300, policy="autonomous")
        res = run_simulation(cfg, tempfile.mkdtemp(prefix="aro_selftest_"), fast=True, render_video=False)
        assert not res["metrics"].empty
        assert "cum_uplift_usd" in res["metrics"]
        assert "summary" in res

    check("duplicate detection", t_duplicate_detection)
    check("missing schema handling", t_missing_schema)
    check("floor rate limiting", t_rate_limit)
    check("low-quality fallback", t_fallback_low_quality)
    check("synthetic record generation", t_synthetic_unique_and_fields)
    check("model smoke test", t_model_smoke)
    check("simulation smoke test", t_simulation_smoke)

    return pd.DataFrame(checks)


PRODUCT_SPEC_MD = """
### Autonomous Ad Revenue Optimizer — prototype spec

**Mission:** recommend and safely simulate auction revenue actions such as floor pricing and demand allocation under guardrails.

**Core objects:** impression, segment key, bid floor, predicted bid value, uncertainty, action, approval, audit event.

**Policies:**
- `static`: transparent baseline.
- `quantile`: rolling segment quantile rule.
- `model`: ML bid-value prediction.
- `autonomous`: expected-revenue optimization with guardrails and approval gates.

**Guardrails:**
- min/max floor,
- max change percent,
- min fill rate,
- uncertainty penalty,
- fraud exclusion,
- consent-aware contextual fallback,
- human approval for consequential changes.

**Evaluation:**
- paired synthetic counterfactual,
- chronological model validation,
- MAE/RMSE/R²,
- 95% interval coverage,
- subgroup error,
- business calculator with user assumptions.

**Safety:**
- synthetic data only,
- no live write path,
- emergency static fallback,
- audit history,
- explainable decision reasons.
"""

# ----------------------------
# Streamlit UI
# ----------------------------

if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="aro_demo_")

with st.sidebar:
    st.header("Control room")
    st.caption("CPU-only synthetic demo. No API keys.")

    scenario = st.selectbox("Scenario", SCENARIOS, index=0)
    policy = st.selectbox("Policy", POLICIES, index=3)
    risk_mode = st.selectbox("Risk mode", RISK_MODES, index=1)

    st.divider()
    st.subheader("Simulation")

    duration_sec = st.slider("Demo seconds", 4, 12, 6, 1)
    fps = st.selectbox("Video FPS", [8, 10, 12], index=1)
    batch_size = st.slider("Impressions per tick", 30, 150, 70, 5)
    history_size = st.slider("History rows", 600, 5000, 2200, 100)

    st.divider()
    st.subheader("Guardrails")

    base_floor_cpm = st.number_input("Base floor CPM ($)", 0.5, 20.0, 4.0, 0.1)
    min_floor_cpm = st.number_input("Min floor CPM ($)", 0.1, 10.0, 0.75, 0.05)
    max_floor_cpm = st.number_input("Max floor CPM ($)", 1.0, 50.0, 28.0, 0.5)

    if min_floor_cpm >= max_floor_cpm:
        max_floor_cpm = min_floor_cpm + 1.0

    max_change_pct = st.slider("Max floor change % per tick", 5, 100, 25, 5)
    exploration_rate = st.slider("Exploration rate", 0.0, 0.25, 0.05, 0.01)
    min_fill_rate = st.slider("Minimum fill rate guardrail", 0.20, 0.95, 0.55, 0.01)
    require_approval_above_pct = st.slider("Human approval threshold % change", 1, 50, 12, 1)
    simulate_human_approval = st.checkbox("Simulate human approval in demo", value=True)
    seed = st.number_input("Random seed", 0, 10000, 42, 1)

    st.divider()
    auto_apply = st.checkbox("Auto-apply sidebar changes", value=False, help="Turn off to avoid heavy reruns while tuning settings.")

    col1, col2 = st.columns(2)
    run_btn = col1.button("Run", type="primary", use_container_width=True)
    reset_btn = col2.button("Reset", use_container_width=True)

    kill_btn = st.button("Emergency static fallback", use_container_width=True)
    clear_kill_btn = st.button("Clear emergency fallback", use_container_width=True)

    if kill_btn:
        st.session_state.kill_switch = True
        st.session_state.force_run = True
    if clear_kill_btn:
        st.session_state.kill_switch = False
        st.session_state.force_run = True

    compare_btn = st.button("Compare policies", use_container_width=True)
    self_test_btn = st.button("Run self-tests", use_container_width=True)

    if self_test_btn:
        with st.spinner("Running self-tests..."):
            st.session_state.self_test_report = run_self_tests()

    if "self_test_report" in st.session_state:
        st.subheader("Self-test report")
        st.dataframe(st.session_state.self_test_report, use_container_width=True)

force_run = st.session_state.pop("force_run", False)

if reset_btn:
    for k in ["result", "comparison", "last_config_hash", "kill_switch", "self_test_report", "run_error"]:
        st.session_state.pop(k, None)
    st.rerun()

config = OptimizerConfig(
    seed=int(seed),
    duration_sec=int(duration_sec),
    fps=int(fps),
    batch_size=int(batch_size),
    history_size=int(history_size),
    base_floor_cpm=float(base_floor_cpm),
    min_floor_cpm=float(min_floor_cpm),
    max_floor_cpm=float(max_floor_cpm),
    max_change_pct=float(max_change_pct),
    exploration_rate=float(exploration_rate),
    min_fill_rate=float(min_fill_rate),
    require_approval_above_pct=float(require_approval_above_pct),
    risk_mode=str(risk_mode),
    policy=str(policy),
    scenario=str(scenario),
    simulate_human_approval=bool(simulate_human_approval),
)

if st.session_state.get("kill_switch"):
    config = replace_config(config, policy="static", risk_mode="conservative", simulate_human_approval=False)

config_hash = hashlib.md5(json.dumps(as_dict(config), sort_keys=True, default=str).encode()).hexdigest()

settings_changed = st.session_state.get("last_config_hash") != config_hash
should_run = ("result" not in st.session_state) or run_btn or force_run or (auto_apply and settings_changed)

if should_run:
    try:
        with st.spinner("Generating synthetic auction history, validating data, training model, simulating policy, and encoding demo video..."):
            st.session_state.result = run_simulation(config, st.session_state.workdir, fast=False, render_video=True)
            st.session_state.last_config_hash = config_hash
            st.session_state.pop("run_error", None)
    except Exception as exc:
        st.session_state.run_error = str(exc)

if "run_error" in st.session_state and "result" not in st.session_state:
    st.error(f"Simulation failed: {st.session_state.run_error}")
    st.stop()

result = st.session_state.get("result")
if result is None:
    st.warning("No simulation result yet. Press Run.")
    st.stop()

summary = result["summary"]
metrics = result["metrics"]
impressions = result["impressions"]
segment_stats = result["segment_stats"]
audit = result["audit"]
model = result["model"]

st.title("🎥 Autonomous Ad Revenue Optimizer — Live Synthetic Demo")
st.caption(SYNTHETIC_NOTICE)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Scenario", summary["scenario"])
c2.metric("Policy", summary["policy"])
c3.metric("Risk", summary["risk_mode"])
c4.metric("Model", "trained" if summary["model_trained"] else "fallback")
c5.metric("Video", "MP4" if result.get("mp4_path") else ("GIF" if result.get("gif_path") else ("PNG" if result.get("final_frame_png") else "none")))
c6.metric("Config hash", config_hash[:10])

if settings_changed and not auto_apply:
    st.info("Sidebar settings changed. Press **Run** to apply them.")

if st.session_state.get("kill_switch"):
    st.warning("Emergency fallback active: static floors and no autonomous changes.")

tab_video, tab_revenue, tab_decisions, tab_model, tab_contract, tab_business, tab_docs = st.tabs(
    ["Live demo video", "Revenue explorer", "Decisions & audit", "Model & risk", "Data contract", "Business case", "Acceptance & docs"]
)

with tab_video:
    st.subheader("Auto-generated demo clip")
    mp4_bytes = None
    gif_bytes = None

    if result.get("mp4_path") and os.path.exists(result["mp4_path"]):
        with open(result["mp4_path"], "rb") as f:
            mp4_bytes = f.read()
    if result.get("gif_path") and os.path.exists(result["gif_path"]):
        with open(result["gif_path"], "rb") as f:
            gif_bytes = f.read()

    if mp4_bytes:
        st.video(mp4_bytes, format="video/mp4", loop=True, autoplay=True, muted=True)
        st.download_button(
            "Download MP4 demo video",
            data=mp4_bytes,
            file_name="autonomous_ad_optimizer_demo.mp4",
            mime="video/mp4",
        )
    elif gif_bytes:
        st.image(gif_bytes, caption="Animated GIF fallback", use_container_width=True)
        st.download_button(
            "Download GIF demo clip",
            data=gif_bytes,
            file_name="autonomous_ad_optimizer_demo.gif",
            mime="image/gif",
        )
    elif result.get("final_frame_png"):
        st.image(result["final_frame_png"], caption="Final frame fallback", use_container_width=True)
        st.download_button(
            "Download final frame PNG",
            data=result["final_frame_png"],
            file_name="autonomous_ad_optimizer_final_frame.png",
            mime="image/png",
        )
    else:
        st.warning("Video encoding was unavailable in this environment. The simulation still ran and tables below are populated.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulated revenue", fmt_usd(summary["cum_revenue"]), f"{summary['uplift_pct']:.1f}% vs baseline")
    m2.metric("Incremental uplift", fmt_usd(summary["uplift_usd"]))
    m3.metric("Fill rate", fmt_pct(summary["fill_rate"]), f"target {fmt_pct(config.min_fill_rate)}")
    m4.metric("Avg floor", f"${summary['avg_floor_cpm']:.2f}")
    m5.metric("Data quality", fmt_pct(summary["data_quality"]))

    st.subheader("Scenario narrative")
    st.write(SCENARIO_DESCRIPTIONS.get(config.scenario, "Custom scenario."))

    st.subheader("Guardrail monitor")
    guardrail_df = pd.DataFrame(
        [
            {"metric": "Cumulative fallbacks", "value": summary["fallback_count"], "interpretation": "Deterministic fallback activations."},
            {"metric": "Pending approvals", "value": summary["pending_approvals"], "interpretation": "Actions held for human review."},
            {"metric": "Simulated approvals", "value": summary["simulated_approvals"], "interpretation": "Consequential actions approved in demo mode."},
            {"metric": "Rate-limited changes", "value": summary["rate_limited_count"], "interpretation": "Floor changes clipped by max step guardrail."},
            {"metric": "Exploration actions", "value": summary["exploration_count"], "interpretation": "Bounded alternative-floor probes."},
            {"metric": "Guardrail violations", "value": summary["guardrail_violations"], "interpretation": "Fill-rate or floor-bound breaches detected."},
        ]
    )
    st.dataframe(guardrail_df, use_container_width=True)

with tab_revenue:
    st.subheader("Cumulative revenue: optimizer vs static baseline")
    if not metrics.empty:
        chart_df = metrics.set_index("step")[["cum_optimizer_revenue_usd", "cum_baseline_revenue_usd"]]
        st.line_chart(chart_df)

        st.subheader("Per-tick uplift")
        st.bar_chart(metrics.set_index("step")[["uplift_usd"]])

        st.subheader("Fill and floor trend")
        trend = metrics.set_index("step")[["fill_rate", "baseline_fill_rate", "avg_floor_cpm"]]
        st.line_chart(trend)
    else:
        st.write("No metrics available.")

    st.subheader("Segment drill-down")
    if not impressions.empty:
        colA, colB, colC, colD = st.columns(4)
        country_opts = unique_strings(impressions["country"])
        device_opts = unique_strings(impressions["device"])
        segment_opts = unique_strings(impressions["audience_segment"])
        demand_opts = unique_strings(impressions["demand_source"])

        sel_countries = colA.multiselect("Country", country_opts, default=country_opts)
        sel_devices = colB.multiselect("Device", device_opts, default=device_opts)
        sel_segments = colC.multiselect("Audience segment", segment_opts, default=segment_opts)
        sel_demand = colD.multiselect("Demand source", demand_opts, default=demand_opts)

        filtered = impressions.copy()
        if sel_countries:
            filtered = filtered[filtered["country"].isin(sel_countries)]
        if sel_devices:
            filtered = filtered[filtered["device"].isin(sel_devices)]
        if sel_segments:
            filtered = filtered[filtered["audience_segment"].isin(sel_segments)]
        if sel_demand:
            filtered = filtered[filtered["demand_source"].isin(sel_demand)]

        if filtered.empty:
            st.warning("No rows match filters.")
        else:
            seg = (
                filtered.groupby("segment_key")
                .agg(
                    impressions=("revenue_cpm", "size"),
                    optimizer_revenue_cpm_sum=("revenue_cpm", "sum"),
                    baseline_revenue_cpm_sum=("baseline_revenue_cpm", "sum"),
                    fill_rate=("cleared", "mean"),
                    avg_floor_cpm=("bid_floor_cpm", "mean"),
                    avg_predicted_bid=("predicted_max_bid_cpm", "mean"),
                    avg_uncertainty=("prediction_uncertainty_cpm", "mean"),
                )
                .reset_index()
            )
            seg["optimizer_revenue_usd"] = seg["optimizer_revenue_cpm_sum"] / 1000.0
            seg["baseline_revenue_usd"] = seg["baseline_revenue_cpm_sum"] / 1000.0
            seg["uplift_usd"] = seg["optimizer_revenue_usd"] - seg["baseline_revenue_usd"]
            seg["uplift_pct"] = seg["uplift_usd"] / seg["baseline_revenue_usd"].replace(0, np.nan) * 100.0
            seg = seg.sort_values("uplift_usd", ascending=False)
            st.dataframe(seg.head(50), use_container_width=True)

            st.download_button(
                "Download filtered impressions CSV",
                data=df_to_csv_bytes(filtered),
                file_name="filtered_impressions.csv",
                mime="text/csv",
            )
    else:
        st.write("No impression rows available.")

    if "comparison" in st.session_state:
        st.subheader("Policy comparison")
        st.dataframe(st.session_state["comparison"], use_container_width=True)
        st.download_button(
            "Download policy comparison CSV",
            data=df_to_csv_bytes(st.session_state["comparison"]),
            file_name="policy_comparison.csv",
            mime="text/csv",
        )

    if compare_btn:
        with st.spinner("Running short comparison across static, quantile, model, and autonomous policies..."):
            st.session_state["comparison"] = compare_policies(config, st.session_state.workdir)
        st.success("Policy comparison complete. See Revenue explorer tab.")

with tab_decisions:
    st.subheader("Auction decisions and approval state")
    if impressions.empty:
        st.write("No decisions available.")
    else:
        status_options = sorted([s for s in impressions["approval_status"].dropna().unique() if str(s)])
        selected_status = st.multiselect("Filter by approval status", status_options, default=status_options)
        filtered = impressions[impressions["approval_status"].isin(selected_status)] if selected_status else impressions

        display_cols = [
            "event_time",
            "segment_key",
            "bid_floor_cpm",
            "predicted_max_bid_cpm",
            "prediction_uncertainty_cpm",
            "decision_reason",
            "approval_status",
            "cleared",
            "revenue_cpm",
            "baseline_revenue_cpm",
        ]
        st.dataframe(filtered[display_cols].tail(500), use_container_width=True)

        st.download_button(
            "Download decisions CSV",
            data=df_to_csv_bytes(filtered[display_cols]),
            file_name="decisions.csv",
            mime="text/csv",
        )

        if not filtered.empty:
            st.subheader("Evidence inspector")
            uid = st.selectbox("Inspect impression", filtered["impression_uid"].astype(str).tolist()[:500])
            row = filtered[filtered["impression_uid"].astype(str) == uid].iloc[0]
            st.json({k: jsonable(v) for k, v in row.items()})

    st.subheader("Audit history")
    audit_df = pd.DataFrame(audit)
    if not audit_df.empty:
        st.dataframe(audit_df.tail(300), use_container_width=True)
        st.download_button(
            "Download audit JSON",
            data=json_bytes(audit),
            file_name="audit_history.json",
            mime="application/json",
        )
    else:
        st.write("No audit events.")

with tab_model:
    st.subheader("Model status")
    if model.trained:
        st.success(f"Model trained: {model.version}")
    else:
        st.warning("Model not trained. Deterministic fallback policy is active.")

    st.json(model.val_metrics)

    if model.val_data is not None and model.val_pred is not None and model.val_unc is not None:
        vd = model.val_data.copy()
        vd["predicted_max_bid_cpm"] = model.val_pred
        vd["prediction_uncertainty_cpm"] = model.val_unc
        vd["abs_error"] = np.abs(vd["max_bid_cpm"] - vd["predicted_max_bid_cpm"])

        st.subheader("Subgroup error")
        for dim in ["country", "device", "consent_state"]:
            if dim in vd:
                grp = vd.groupby(dim)["abs_error"].agg(["mean", "count"]).reset_index()
                grp = grp.rename(columns={"mean": "MAE", "count": "validation_rows"})
                st.write(f"**By {dim}**")
                st.dataframe(grp, use_container_width=True)

        coverage = float(np.mean(vd["abs_error"] <= 1.96 * vd["prediction_uncertainty_cpm"]))
        st.metric("Validation 95% interval coverage", fmt_pct(coverage))
    else:
        st.info("No validation subset available.")

    st.subheader("Feature list")
    st.write(model.feature_names if model.feature_names else ["fallback rule only"])

    st.subheader("Risk register")
    st.dataframe(pd.DataFrame(RISK_REGISTER), use_container_width=True)

    st.subheader("Actions requiring human approval")
    st.markdown(
        """
- enabling full autonomy in production,
- floor changes above configured threshold,
- expanding targeting to new audience categories,
- changing privacy-sensitive feature usage,
- disabling guardrails,
- rolling out to high-revenue publisher segments,
- overriding fraud or consent warnings.
        """
    )

with tab_contract:
    st.subheader("Field dictionary")
    st.dataframe(pd.DataFrame(FIELD_DICTIONARY), use_container_width=True)

    st.subheader("Example synthetic record")
    example = {
        "impression_uid": "7c8f9d2e-5c54-4d54-9c0d-5c9b1d2f3a44",
        "event_time": "2026-11-06T14:03:21+00:00",
        "publisher_id": "pub_newsnet",
        "ad_unit": "mid_article",
        "country": "US",
        "device": "mobile",
        "audience_segment": "finance",
        "demand_source": "dsp_alpha",
        "consent_state": True,
        "fraud_flag": False,
        "bid_floor_cpm": 6.12,
        "max_bid_cpm": 8.44,
        "second_bid_cpm": 6.02,
        "winning_bid_cpm": 6.12,
        "revenue_cpm": 6.12,
        "cleared": True,
        "latency_ms": 21.4,
        "segment_key": "US|mobile|finance|dsp_alpha",
        "predicted_max_bid_cpm": 8.10,
        "prediction_uncertainty_cpm": 0.95,
        "decision_reason": "expected_revenue_optimizer; rate_limited",
        "approval_status": "auto_within_guardrail",
        "baseline_revenue_cpm": 4.00,
        "model_version": "hgb-bid-ensemble-v2-0.412",
        "experiment_arm": "treat_a",
    }
    st.json(example)

    st.subheader("Handling rules")
    st.dataframe(pd.DataFrame(HANDLING_RULES), use_container_width=True)

with tab_business:
    st.subheader("Business value calculator (user-entered assumptions)")
    colA, colB, colC = st.columns(3)
    monthly_impressions = colA.number_input(
        "Monthly billable impressions",
        min_value=100_000,
        max_value=10_000_000_000,
        value=50_000_000,
        step=1_000_000,
    )
    current_rpm = colB.number_input(
        "Current RPM ($ per 1000 impressions)",
        min_value=0.10,
        max_value=100.00,
        value=4.20,
        step=0.10,
    )
    uplift_pct_input = colC.number_input(
        "Simulated uplift % to model",
        min_value=0.0,
        max_value=100.0,
        value=max(0.0, safe_float(summary.get("uplift_pct", 0.0))),
        step=0.1,
    )

    colD, colE = st.columns(2)
    monthly_op_cost = colD.number_input("Monthly operating cost ($)", min_value=0.0, max_value=1_000_000.0, value=2_500.0, step=100.0)
    one_time_setup = colE.number_input("One-time setup/integration cost ($)", min_value=0.0, max_value=1_000_000.0, value=15_000.0, step=500.0)

    gross_monthly = float(monthly_impressions) / 1000.0 * float(current_rpm) * float(uplift_pct_input) / 100.0
    net_monthly = gross_monthly - float(monthly_op_cost)
    payback_months = (float(one_time_setup) / net_monthly) if net_monthly > 0 else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Gross incremental revenue / month", fmt_usd(gross_monthly))
    m2.metric("Net after operating cost / month", fmt_usd(net_monthly))
    m3.metric("Payback period", f"{payback_months:.1f} months" if payback_months else "Not positive under assumptions")

    sensitivity = []
    for uplift_mult in [0.5, 1.0, 1.5]:
        for cost_mult in [0.75, 1.0, 1.25]:
            g = float(monthly_impressions) / 1000.0 * float(current_rpm) * float(uplift_pct_input) / 100.0 * uplift_mult
            n = g - float(monthly_op_cost) * cost_mult
            sensitivity.append(
                {
                    "uplift_multiplier": uplift_mult,
                    "cost_multiplier": cost_mult,
                    "gross_monthly_usd": g,
                    "net_monthly_usd": n,
                    "payback_months": (float(one_time_setup) / n) if n > 0 else None,
                }
            )
    st.dataframe(pd.DataFrame(sensitivity), use_container_width=True)
    st.caption(
        "This calculator uses user-entered assumptions and the simulated uplift from the synthetic demo. "
        "It does not fabricate customer results or guarantee commercial performance."
    )

with tab_docs:
    st.markdown(PRODUCT_SPEC_MD)

    st.subheader("Acceptance checks")
    st.dataframe(pd.DataFrame(ACCEPTANCE_CHECKS), use_container_width=True)

    st.subheader("Demo script")
    st.markdown(
        """
1. Launch app: `streamlit run app.py`
2. Wait for auto-generated demo video.
3. Show cumulative revenue, uplift, fill rate, data quality.
4. Switch scenario to `schema_drift` and press Run.
5. Show validator warning and fallback behavior.
6. Switch scenario to `model_drift` and press Run.
7. Show drift monitor and conservative response.
8. Open Decisions & audit and inspect one recommendation.
9. Open Model & risk and show subgroup error and coverage.
10. Open Business case and explain assumptions are user-entered.
        """
    )

    st.subheader("Known limitations")
    st.markdown(
        """
- Synthetic data only.
- No real ad-server integration.
- No production causal inference.
- No enterprise RBAC or SSO.
- Video encoding depends on local environment; GIF/PNG fallback is included.
- Model is intentionally simple for CPU reproducibility.
        """
    )

    st.subheader("Roadmap")
    st.dataframe(pd.DataFrame(ROADMAP), use_container_width=True)

    st.subheader("License inventory")
    st.dataframe(pd.DataFrame(LICENSE_INVENTORY), use_container_width=True)

    st.subheader("Reproducibility")
    st.json(
        {
            "app_version": APP_VERSION,
            "config_hash": config_hash,
            "seed": config.seed,
            "scenario": config.scenario,
            "policy": config.policy,
            "risk_mode": config.risk_mode,
            "model_version": summary["model_version"],
        }
    )