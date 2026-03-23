# -*- coding: utf-8 -*-
"""
Predictive SPC Streamlit App (Demo Data Included)
------------------------------------------------
What this app does
- Generates realistic demo manufacturing/SPC data for multiple part numbers and machines
- Builds forward-looking prediction targets:
    1) Out-of-spec risk in the next N samples
    2) Nelson-style rule risk in the next N samples
- Trains tree-based ML models on engineered SPC/process features
- Shows current risk, likely drivers, prediction charts, and recommended actions
- Includes a second dashboard tab showing current status by part/machine combo
  for a selected date/time period
- Includes colored status tiles for quick glance viewing

Run:
    streamlit run pred_spcv2.py
"""

import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Predictive SPC Demo", layout="wide")


# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.title("📈 Predictive SPC Demo App")
st.caption(
    "Forward-looking SPC with demo data: predict out-of-spec risk and future rule risk before violations happen."
)

st.sidebar.info(
    "**Questions or Issues?**\n\n"
    "📧 Contact: [chad@hertzler.com](mailto:chad@hertzler.com)"
)


# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class ModelBundle:
    feature_cols: List[str]
    model_oos: RandomForestClassifier
    model_rule: RandomForestClassifier
    test_metrics: dict


# -----------------------------------------------------------------------------
# DEMO DATA GENERATION
# -----------------------------------------------------------------------------
def make_demo_data(
    n_parts: int = 5,
    n_machines: int = 3,
    rows_per_combo: int = 650,
    freq_minutes: int = 15,
) -> pd.DataFrame:
    """Generate realistic process and measurement data with drift, shift, and noise."""
    rng = np.random.default_rng(RANDOM_SEED)

    parts = [f"PN-{i+1:03d}" for i in range(n_parts)]
    machines = [f"M{i+1}" for i in range(n_machines)]
    shifts = ["A", "B", "C"]

    start_ts = pd.Timestamp("2025-01-01 06:00:00")
    records = []

    for p_idx, part in enumerate(parts):
        target = 10.0 + (p_idx * 0.85)
        spec_half_width = 0.28 + (p_idx % 3) * 0.05
        lsl = target - spec_half_width
        usl = target + spec_half_width

        base_temp = 190 + p_idx * 7
        base_pressure = 85 + p_idx * 5
        base_speed = 110 + p_idx * 8
        base_humidity = 45 + p_idx * 2
        material_factor = 0.0

        for m_idx, machine in enumerate(machines):
            machine_bias = (m_idx - 1) * 0.045
            wear = 0.0
            level = target + machine_bias
            temp = base_temp + m_idx * 1.5
            pressure = base_pressure + m_idx * 1.8
            speed = base_speed - m_idx * 2.5
            humidity = base_humidity + m_idx * 1.0

            timestamps = pd.date_range(
                start=start_ts,
                periods=rows_per_combo,
                freq=f"{freq_minutes}min",
            )

            drift_dir = rng.choice([-1, 1])
            upcoming_event_remaining = 0

            for i, ts in enumerate(timestamps):
                shift = shifts[(ts.hour // 8) % 3]

                if i % 180 == 0 and i > 0:
                    material_factor = rng.normal(0, 0.02)
                if i % 120 == 0 and i > 0:
                    drift_dir = rng.choice([-1, 1])
                if i % 210 == 0 and i > 0:
                    upcoming_event_remaining = rng.integers(8, 20)

                wear += rng.normal(0.0008, 0.0003)
                if i % 260 == 0 and i > 0:
                    wear = max(0.0, wear - 0.18)

                temp += rng.normal(0, 0.35) + 0.012 * drift_dir + 0.02 * wear
                pressure += rng.normal(0, 0.28) + 0.009 * drift_dir + 0.017 * wear
                speed += rng.normal(0, 0.40) - 0.008 * drift_dir - 0.010 * wear
                humidity += rng.normal(0, 0.18)

                episodic_shift = 0.0
                episodic_var_boost = 1.0
                if upcoming_event_remaining > 0:
                    episodic_shift = drift_dir * (0.015 + 0.002 * (20 - upcoming_event_remaining))
                    episodic_var_boost = 1.4
                    upcoming_event_remaining -= 1

                temp_effect = 0.0042 * (temp - base_temp)
                pressure_effect = 0.0036 * (pressure - base_pressure)
                speed_effect = -0.0028 * (speed - base_speed)
                humidity_effect = 0.0012 * (humidity - base_humidity)
                wear_effect = 0.030 * wear

                level += (0.0021 * drift_dir) + rng.normal(0, 0.006)

                noise_sigma = (0.040 + 0.015 * wear) * episodic_var_boost
                measurement = (
                    level
                    + temp_effect
                    + pressure_effect
                    + speed_effect
                    + humidity_effect
                    + wear_effect
                    + material_factor
                    + episodic_shift
                    + rng.normal(0, noise_sigma)
                )

                records.append(
                    {
                        "timestamp": ts,
                        "part_no": part,
                        "machine": machine,
                        "shift": shift,
                        "measurement": measurement,
                        "target": target,
                        "lsl": lsl,
                        "usl": usl,
                        "temperature": temp,
                        "pressure": pressure,
                        "speed": speed,
                        "humidity": humidity,
                        "wear_index": wear,
                    }
                )

    df = (
        pd.DataFrame(records)
        .sort_values(["part_no", "machine", "timestamp"])
        .reset_index(drop=True)
    )
    return df


# -----------------------------------------------------------------------------
# SPC / FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def rolling_slope(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if len(arr) < 2:
        return 0.0

    x = np.arange(len(arr), dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    slope = np.sum((x - x_mean) * (arr - y_mean)) / denom
    return float(slope)


def consecutive_same_side_run(z_values) -> int:
    if len(z_values) == 0:
        return 0

    signs = [1 if z > 0 else (-1 if z < 0 else 0) for z in z_values]
    last = signs[-1]
    if last == 0:
        return 0

    run = 0
    for s in reversed(signs):
        if s == last:
            run += 1
        else:
            break
    return run


def consecutive_trend_run(values) -> int:
    if len(values) < 2:
        return 0

    diffs = np.diff(values)
    if len(diffs) == 0:
        return 0

    last_dir = 1 if diffs[-1] > 0 else (-1 if diffs[-1] < 0 else 0)
    if last_dir == 0:
        return 0

    run = 1
    for d in diffs[::-1]:
        cur_dir = 1 if d > 0 else (-1 if d < 0 else 0)
        if cur_dir == last_dir:
            run += 1
        else:
            break
    return run


def nelson_like_rule_trigger(window_measurements: pd.Series, center: float, sigma: float) -> int:
    """
    Simplified Nelson-style future event flag.
    Triggers if any of the following hold inside a future window:
      1) Any point beyond 3 sigma
      2) 8 consecutive points on one side of center
      3) 6 consecutive increasing or decreasing points
      4) 2 of 3 beyond 2 sigma on same side
    """
    vals = window_measurements.to_numpy(dtype=float)
    if len(vals) == 0 or sigma <= 1e-12:
        return 0

    z = (vals - center) / sigma

    if np.any(np.abs(z) > 3):
        return 1

    if len(z) >= 8:
        signs = np.sign(z)
        for i in range(len(signs) - 7):
            chunk = signs[i:i + 8]
            if np.all(chunk > 0) or np.all(chunk < 0):
                return 1

    if len(vals) >= 6:
        for i in range(len(vals) - 5):
            chunk = vals[i:i + 6]
            d = np.diff(chunk)
            if np.all(d > 0) or np.all(d < 0):
                return 1

    if len(z) >= 3:
        for i in range(len(z) - 2):
            chunk = z[i:i + 3]
            if np.sum(chunk > 2) >= 2 or np.sum(chunk < -2) >= 2:
                return 1

    return 0


def add_features_and_targets(df: pd.DataFrame, future_horizon: int = 8) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(["part_no", "machine"], sort=False)

    out["deviation"] = out["measurement"] - out["target"]
    out["spec_range"] = out["usl"] - out["lsl"]
    out["dist_to_usl"] = out["usl"] - out["measurement"]
    out["dist_to_lsl"] = out["measurement"] - out["lsl"]
    out["pct_of_spec_used"] = np.abs(out["measurement"] - out["target"]) / (out["spec_range"] / 2)

    for win in [3, 5, 8, 12, 20]:
        out[f"roll_mean_{win}"] = grp["measurement"].transform(
            lambda s: s.rolling(win, min_periods=2).mean()
        )
        out[f"roll_std_{win}"] = grp["measurement"].transform(
            lambda s: s.rolling(win, min_periods=2).std(ddof=0)
        )
        out[f"roll_min_{win}"] = grp["measurement"].transform(
            lambda s: s.rolling(win, min_periods=2).min()
        )
        out[f"roll_max_{win}"] = grp["measurement"].transform(
            lambda s: s.rolling(win, min_periods=2).max()
        )
        out[f"slope_{win}"] = grp["measurement"].transform(
            lambda s: s.rolling(win, min_periods=3).apply(rolling_slope, raw=False)
        )

    out["mr_1"] = grp["measurement"].diff().abs()
    out["mr_roll_8"] = grp["mr_1"].transform(lambda s: s.rolling(8, min_periods=2).mean())
    out["ewma_0_3"] = grp["measurement"].transform(lambda s: s.ewm(alpha=0.30, adjust=False).mean())
    out["ewma_0_1"] = grp["measurement"].transform(lambda s: s.ewm(alpha=0.10, adjust=False).mean())

    out["baseline_center_20"] = grp["measurement"].transform(
        lambda s: s.rolling(20, min_periods=8).mean()
    )
    out["baseline_sigma_20"] = grp["measurement"].transform(
        lambda s: s.rolling(20, min_periods=8).std(ddof=0)
    )
    out["baseline_sigma_20"] = out["baseline_sigma_20"].replace(0, np.nan)
    out["z_from_baseline"] = (out["measurement"] - out["baseline_center_20"]) / out["baseline_sigma_20"]

    out["run_same_side_8"] = grp["z_from_baseline"].transform(
        lambda s: s.rolling(8, min_periods=3).apply(
            lambda x: consecutive_same_side_run(list(x)),
            raw=False,
        )
    )
    out["run_trend_8"] = grp["measurement"].transform(
        lambda s: s.rolling(8, min_periods=3).apply(
            lambda x: consecutive_trend_run(list(x)),
            raw=False,
        )
    )

    out["is_oos_now"] = (
        ((out["measurement"] < out["lsl"]) | (out["measurement"] > out["usl"]))
        .astype(int)
    )
    out["near_spec_warn"] = (
        (out["pct_of_spec_used"] >= 0.80) & (out["pct_of_spec_used"] < 1.0)
    ).astype(int)
    out["recent_oos_8"] = grp["is_oos_now"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    out["recent_oos_20"] = grp["is_oos_now"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    out["recent_warn_8"] = grp["near_spec_warn"].transform(lambda s: s.rolling(8, min_periods=1).sum())

    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    out["row_in_group"] = grp.cumcount()

    original_groups = df.groupby(["part_no", "machine"], sort=False)

    targets_oos = []
    targets_rule = []

    for _, g in original_groups:
        g = g.sort_values("timestamp").reset_index(drop=True)
        center = g["measurement"].rolling(20, min_periods=8).mean()
        sigma = g["measurement"].rolling(20, min_periods=8).std(ddof=0).replace(0, np.nan)

        future_oos = np.full(len(g), np.nan, dtype=float)
        future_rule = np.full(len(g), np.nan, dtype=float)

        for i in range(len(g)):
            future_slice = g.iloc[i + 1:i + 1 + future_horizon]
            if len(future_slice) == 0:
                continue

            future_oos[i] = float(
                (
                    (future_slice["measurement"] < future_slice["lsl"])
                    | (future_slice["measurement"] > future_slice["usl"])
                ).any()
            )

            c = center.iloc[i]
            s = sigma.iloc[i]
            if pd.isna(c) or pd.isna(s) or s <= 1e-12:
                future_rule[i] = np.nan
            else:
                future_rule[i] = float(
                    nelson_like_rule_trigger(future_slice["measurement"], c, s)
                )

        targets_oos.extend(list(future_oos))
        targets_rule.extend(list(future_rule))

    out["target_future_oos"] = targets_oos
    out["target_future_rule"] = targets_rule

    out = pd.get_dummies(out, columns=["shift", "machine", "part_no"], drop_first=False)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c.startswith("target_"):
            continue
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(out[c].median())

    out = out.dropna(subset=["target_future_oos", "target_future_rule"]).copy()
    out["target_future_oos"] = out["target_future_oos"].astype(int)
    out["target_future_rule"] = out["target_future_rule"].astype(int)

    return out


# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------
def train_predictive_models(df_model: pd.DataFrame) -> ModelBundle:
    feature_cols = [
        c
        for c in df_model.columns
        if c not in {
            "timestamp",
            "measurement",
            "lsl",
            "usl",
            "target",
            "target_future_oos",
            "target_future_rule",
        }
        and pd.api.types.is_numeric_dtype(df_model[c])
    ]

    df_model = df_model.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_model) * 0.80)
    train_df = df_model.iloc[:split_idx].copy()
    test_df = df_model.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_oos = train_df["target_future_oos"]
    y_test_oos = test_df["target_future_oos"]

    y_train_rule = train_df["target_future_rule"]
    y_test_rule = test_df["target_future_rule"]

    model_oos = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model_rule = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=8,
        random_state=RANDOM_SEED + 7,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model_oos.fit(X_train, y_train_oos)
    model_rule.fit(X_train, y_train_rule)

    pred_oos = model_oos.predict(X_test)
    pred_rule = model_rule.predict(X_test)
    prob_oos = model_oos.predict_proba(X_test)[:, 1]
    prob_rule = model_rule.predict_proba(X_test)[:, 1]

    metrics = {
        "oos_accuracy": accuracy_score(y_test_oos, pred_oos),
        "oos_auc": roc_auc_score(y_test_oos, prob_oos) if len(np.unique(y_test_oos)) > 1 else np.nan,
        "rule_accuracy": accuracy_score(y_test_rule, pred_rule),
        "rule_auc": roc_auc_score(y_test_rule, prob_rule) if len(np.unique(y_test_rule)) > 1 else np.nan,
        "oos_classification_report": classification_report(y_test_oos, pred_oos, output_dict=False, zero_division=0),
        "rule_classification_report": classification_report(y_test_rule, pred_rule, output_dict=False, zero_division=0),
    }

    return ModelBundle(
        feature_cols=feature_cols,
        model_oos=model_oos,
        model_rule=model_rule,
        test_metrics=metrics,
    )


# -----------------------------------------------------------------------------
# SCORING / ACTIONS / UI HELPERS
# -----------------------------------------------------------------------------
def probability_to_label(p: float) -> str:
    if p >= 0.80:
        return "Critical"
    if p >= 0.60:
        return "High"
    if p >= 0.35:
        return "Moderate"
    return "Low"


def make_action_text(oos_prob: float, rule_prob: float, slope: float, pct_spec: float) -> str:
    actions = []

    if oos_prob >= 0.80:
        actions.append("Increase inspection frequency immediately and verify current setup.")
    elif oos_prob >= 0.60:
        actions.append("Temporarily tighten inspection interval and review recent parameter shifts.")

    if rule_prob >= 0.75:
        actions.append("Investigate process drift before control-chart violations appear.")

    if slope > 0.01:
        actions.append("Process mean is drifting upward. Check heat, pressure, or tool condition.")
    elif slope < -0.01:
        actions.append("Process mean is drifting downward. Check speed, material, or setup bias.")

    if pct_spec >= 0.90:
        actions.append("Process is close to a spec edge. Confirm centering around target.")

    if not actions:
        return "Process looks stable right now. Maintain standard inspection and keep monitoring the risk trend."

    return " ".join(actions)


def top_feature_table(model: RandomForestClassifier, feature_cols: List[str], top_n: int = 12) -> pd.DataFrame:
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return fi.head(top_n).reset_index(drop=True)


def local_driver_table(
    row_features: pd.Series,
    model: RandomForestClassifier,
    feature_cols: List[str],
    top_n: int = 8,
) -> pd.DataFrame:
    x = row_features[feature_cols].astype(float)
    score = np.abs(x.to_numpy()) * model.feature_importances_
    d = pd.DataFrame({"feature": feature_cols, "driver_score": score})
    d = d.sort_values("driver_score", ascending=False).head(top_n).reset_index(drop=True)
    return d


def make_spc_chart(df_plot: pd.DataFrame, selected_idx: int) -> go.Figure:
    mu = df_plot["measurement"].rolling(20, min_periods=8).mean()
    sigma = df_plot["measurement"].rolling(20, min_periods=8).std(ddof=0)
    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_plot["timestamp"], y=df_plot["measurement"], mode="lines+markers", name="Measurement")
    )
    fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot["target"], mode="lines", name="Target"))
    fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot["usl"], mode="lines", name="USL"))
    fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot["lsl"], mode="lines", name="LSL"))
    fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=ucl, mode="lines", name="Rolling UCL"))
    fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=lcl, mode="lines", name="Rolling LCL"))

    if 0 <= selected_idx < len(df_plot):
        sel_row = df_plot.iloc[selected_idx]
        fig.add_vline(x=sel_row["timestamp"], line_width=2, line_dash="dash")

    fig.update_layout(
        title="Measurement with Specs and Rolling Control Limits",
        xaxis_title="Timestamp",
        yaxis_title="Measurement",
        height=520,
        legend_title="Series",
    )
    return fig


def make_risk_chart(df_plot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_plot["timestamp"], y=df_plot["pred_prob_oos"], mode="lines", name="Future OOS Risk")
    )
    fig.add_trace(
        go.Scatter(x=df_plot["timestamp"], y=df_plot["pred_prob_rule"], mode="lines", name="Future Rule Risk")
    )
    fig.update_layout(
        title="Forward Risk Trend",
        xaxis_title="Timestamp",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
    )
    return fig


def make_status_tile_html(row) -> str:
    status = row["current_status"]

    color_map = {
        "Critical": "#d9534f",
        "High": "#f0ad4e",
        "Moderate": "#ffd966",
        "Low": "#5cb85c",
    }
    bg = color_map.get(status, "#cccccc")
    text_color = "#000000" if status == "Moderate" else "#ffffff"

    oos_now = "Yes" if int(row["is_oos_now"]) == 1 else "No"

    if status == "Critical":
        action = "Act now"
    elif status == "High":
        action = "Investigate soon"
    elif status == "Moderate":
        action = "Watch closely"
    else:
        action = "Stable"

    html = f"""
    <div style="
        background-color:{bg};
        color:{text_color};
        padding:14px;
        border-radius:12px;
        min-height:220px;
        box-shadow:0 2px 6px rgba(0,0,0,0.15);
        margin-bottom:12px;
    ">
        <div style="font-size:18px; font-weight:700; margin-bottom:6px;">
            {row['part_display']} | {row['machine_display']}
        </div>
        <div style="font-size:13px; margin-bottom:8px;">
            <b>Status:</b> {row['current_status']}<br>
            <b>Latest:</b> {row['timestamp']}<br>
            <b>Measurement:</b> {row['measurement']:.4f}<br>
            <b>Target:</b> {row['target']:.4f}<br>
            <b>OOS Now:</b> {oos_now}
        </div>
        <div style="font-size:13px; margin-bottom:8px;">
            <b>Future OOS Risk:</b> {row['pred_prob_oos']:.1%}<br>
            <b>Future Rule Risk:</b> {row['pred_prob_rule']:.1%}<br>
            <b>Spec Usage:</b> {row['pct_of_spec_used']:.1%}
        </div>
        <div style="font-size:14px; font-weight:700;">
            {action}
        </div>
    </div>
    """
    return html


# -----------------------------------------------------------------------------
# DATA PREP + MODEL TRAINING CACHE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_demo_modeling_data(
    n_parts: int,
    n_machines: int,
    rows_per_combo: int,
    future_horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = make_demo_data(
        n_parts=n_parts,
        n_machines=n_machines,
        rows_per_combo=rows_per_combo,
    )
    model_df = add_features_and_targets(raw, future_horizon=future_horizon)
    return raw, model_df


@st.cache_resource(show_spinner=False)
def get_trained_models(model_df: pd.DataFrame) -> ModelBundle:
    return train_predictive_models(model_df)


# -----------------------------------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------------------------------
st.sidebar.header("Demo Settings")
future_horizon = st.sidebar.slider("Future prediction horizon (samples)", 3, 20, 8, 1)
n_parts = st.sidebar.slider("Number of demo parts", 3, 8, 5, 1)
n_machines = st.sidebar.slider("Number of demo machines", 2, 5, 3, 1)
rows_per_combo = st.sidebar.slider("Rows per part-machine", 300, 1200, 650, 50)

with st.spinner("Generating demo data and training models..."):
    raw_df, model_df = get_demo_modeling_data(
        n_parts=n_parts,
        n_machines=n_machines,
        rows_per_combo=rows_per_combo,
        future_horizon=future_horizon,
    )
    bundle = get_trained_models(model_df)


# -----------------------------------------------------------------------------
# SCORE ENTIRE MODEL DATASET FOR UI
# -----------------------------------------------------------------------------
scored_df = model_df.copy()
scored_df["pred_prob_oos"] = bundle.model_oos.predict_proba(scored_df[bundle.feature_cols])[:, 1]
scored_df["pred_prob_rule"] = bundle.model_rule.predict_proba(scored_df[bundle.feature_cols])[:, 1]
scored_df["risk_label_oos"] = scored_df["pred_prob_oos"].map(probability_to_label)
scored_df["risk_label_rule"] = scored_df["pred_prob_rule"].map(probability_to_label)


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔎 Detail View", "📊 Current Status Dashboard"])


# -----------------------------------------------------------------------------
# DETAIL VIEW TAB
# -----------------------------------------------------------------------------
with tab1:
    part_options = sorted(raw_df["part_no"].unique().tolist())
    machine_options = sorted(raw_df["machine"].unique().tolist())

    colf1, colf2 = st.columns([1, 1])
    with colf1:
        selected_part = st.selectbox("Select Part Number", options=part_options, index=0)
    with colf2:
        selected_machine = st.selectbox("Select Machine", options=machine_options, index=0)

    part_col = f"part_no_{selected_part}"
    machine_col = f"machine_{selected_machine}"

    view_df = (
        scored_df[(scored_df[part_col] == 1) & (scored_df[machine_col] == 1)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if view_df.empty:
        st.error("No data available for that selection.")
        st.stop()

    selected_idx = st.slider(
        "Choose current sample index",
        min_value=0,
        max_value=len(view_df) - 1,
        value=len(view_df) - 1,
    )
    current_row = view_df.iloc[selected_idx]

    oos_prob = float(current_row["pred_prob_oos"])
    rule_prob = float(current_row["pred_prob_rule"])
    cur_slope = float(current_row.get("slope_8", 0.0))
    pct_spec = float(current_row.get("pct_of_spec_used", 0.0))

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Future OOS Risk", f"{oos_prob:.1%}", probability_to_label(oos_prob))
    mc2.metric("Future Rule Risk", f"{rule_prob:.1%}", probability_to_label(rule_prob))
    mc3.metric("Current Spec Usage", f"{pct_spec:.1%}")
    mc4.metric("Recent Slope (8)", f"{cur_slope:.4f}")

    st.markdown(
        f"**Recommended Action:** {make_action_text(oos_prob, rule_prob, cur_slope, pct_spec)}"
    )

    left, right = st.columns([1.4, 1.0])
    with left:
        st.plotly_chart(make_spc_chart(view_df, selected_idx), use_container_width=True)
    with right:
        st.plotly_chart(make_risk_chart(view_df), use_container_width=True)

    st.subheader("Current Sample Snapshot")
    info_cols = st.columns(5)
    info_cols[0].write(f"**Timestamp:** {current_row['timestamp']}")
    info_cols[1].write(f"**Measurement:** {current_row['measurement']:.4f}")
    info_cols[2].write(f"**Target:** {current_row['target']:.4f}")
    info_cols[3].write(f"**LSL / USL:** {current_row['lsl']:.4f} / {current_row['usl']:.4f}")
    info_cols[4].write(f"**Currently OOS:** {'Yes' if int(current_row['is_oos_now']) == 1 else 'No'}")

    proc_cols = st.columns(5)
    proc_cols[0].write(f"**Temperature:** {current_row['temperature']:.2f}")
    proc_cols[1].write(f"**Pressure:** {current_row['pressure']:.2f}")
    proc_cols[2].write(f"**Speed:** {current_row['speed']:.2f}")
    proc_cols[3].write(f"**Humidity:** {current_row['humidity']:.2f}")
    proc_cols[4].write(f"**Wear Index:** {current_row['wear_index']:.3f}")

    st.subheader("Likely Drivers")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Local Drivers for Future OOS Risk**")
        st.dataframe(
            local_driver_table(current_row, bundle.model_oos, bundle.feature_cols, top_n=8),
            use_container_width=True,
            hide_index=True,
        )
    with d2:
        st.markdown("**Local Drivers for Future Rule Risk**")
        st.dataframe(
            local_driver_table(current_row, bundle.model_rule, bundle.feature_cols, top_n=8),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Global Feature Importance"):
        gi1, gi2 = st.columns(2)
        with gi1:
            st.markdown("**Global Importance - OOS Model**")
            st.dataframe(
                top_feature_table(bundle.model_oos, bundle.feature_cols),
                use_container_width=True,
                hide_index=True,
            )
        with gi2:
            st.markdown("**Global Importance - Rule Model**")
            st.dataframe(
                top_feature_table(bundle.model_rule, bundle.feature_cols),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Model Performance (Demo Data Test Set)"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OOS Accuracy", f"{bundle.test_metrics['oos_accuracy']:.3f}")
        m2.metric(
            "OOS ROC AUC",
            f"{bundle.test_metrics['oos_auc']:.3f}" if pd.notna(bundle.test_metrics["oos_auc"]) else "N/A",
        )
        m3.metric("Rule Accuracy", f"{bundle.test_metrics['rule_accuracy']:.3f}")
        m4.metric(
            "Rule ROC AUC",
            f"{bundle.test_metrics['rule_auc']:.3f}" if pd.notna(bundle.test_metrics["rule_auc"]) else "N/A",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.text("OOS Classification Report")
            st.code(bundle.test_metrics["oos_classification_report"])
        with c2:
            st.text("Rule Classification Report")
            st.code(bundle.test_metrics["rule_classification_report"])

    with st.expander("Preview Scored Data"):
        preview_cols = [
            "timestamp",
            "measurement",
            "target",
            "lsl",
            "usl",
            "temperature",
            "pressure",
            "speed",
            "humidity",
            "wear_index",
            "pred_prob_oos",
            "pred_prob_rule",
            "target_future_oos",
            "target_future_rule",
            "pct_of_spec_used",
            "slope_8",
            "recent_oos_8",
            "recent_warn_8",
        ]
        st.dataframe(view_df[preview_cols].tail(100), use_container_width=True, hide_index=True)

    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Selected Part/Machine Scored Data as CSV",
        data=csv_bytes,
        file_name=f"predictive_spc_{selected_part}_{selected_machine}.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# CURRENT STATUS DASHBOARD TAB
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Current Status by Part / Machine")

    min_ts = pd.to_datetime(scored_df["timestamp"].min()).to_pydatetime()
    max_ts = pd.to_datetime(scored_df["timestamp"].max()).to_pydatetime()

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        start_date = st.date_input("Start date", value=min_ts.date(), key="start_date")
        start_time = st.time_input("Start time", value=min_ts.time(), key="start_time")

    with dcol2:
        end_date = st.date_input("End date", value=max_ts.date(), key="end_date")
        end_time = st.time_input("End time", value=max_ts.time(), key="end_time")

    start_dt = pd.Timestamp(dt.datetime.combine(start_date, start_time))
    end_dt = pd.Timestamp(dt.datetime.combine(end_date, end_time))

    if start_dt > end_dt:
        st.error("Start date/time must be earlier than or equal to end date/time.")
    else:
        dash_df = scored_df[
            (scored_df["timestamp"] >= start_dt) & (scored_df["timestamp"] <= end_dt)
        ].copy()

        if dash_df.empty:
            st.warning("No records found for the selected date/time period.")
        else:
            part_indicator_cols = [c for c in dash_df.columns if c.startswith("part_no_")]
            machine_indicator_cols = [c for c in dash_df.columns if c.startswith("machine_")]

            dash_df["part_display"] = (
                dash_df[part_indicator_cols].idxmax(axis=1).str.replace("part_no_", "", regex=False)
            )
            dash_df["machine_display"] = (
                dash_df[machine_indicator_cols].idxmax(axis=1).str.replace("machine_", "", regex=False)
            )

            latest_idx = dash_df.groupby(["part_display", "machine_display"])["timestamp"].idxmax()
            status_df = (
                dash_df.loc[latest_idx]
                .copy()
                .sort_values(["part_display", "machine_display"])
                .reset_index(drop=True)
            )

            status_df["current_status"] = np.select(
                [
                    (status_df["pred_prob_oos"] >= 0.80) | (status_df["pred_prob_rule"] >= 0.80),
                    (status_df["pred_prob_oos"] >= 0.60) | (status_df["pred_prob_rule"] >= 0.60),
                    (status_df["pred_prob_oos"] >= 0.35) | (status_df["pred_prob_rule"] >= 0.35),
                ],
                ["Critical", "High", "Moderate"],
                default="Low",
            )

            summary1, summary2, summary3, summary4 = st.columns(4)
            summary1.metric("Combos in Period", f"{status_df.shape[0]}")
            summary2.metric("Critical Combos", f"{(status_df['current_status'] == 'Critical').sum()}")
            summary3.metric("High Combos", f"{(status_df['current_status'] == 'High').sum()}")
            summary4.metric("Moderate Combos", f"{(status_df['current_status'] == 'Moderate').sum()}")

            st.markdown("### Visual Status Board")

            status_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
            status_df["status_rank"] = status_df["current_status"].map(status_order)
            status_df = status_df.sort_values(
                ["status_rank", "part_display", "machine_display"]
            ).reset_index(drop=True)

            cards_per_row = 4
            for i in range(0, len(status_df), cards_per_row):
                row_slice = status_df.iloc[i:i + cards_per_row]
                cols = st.columns(cards_per_row)

                for j, (_, card_row) in enumerate(row_slice.iterrows()):
                    with cols[j]:
                        st.markdown(make_status_tile_html(card_row), unsafe_allow_html=True)

            show_cols = [
                "part_display",
                "machine_display",
                "timestamp",
                "measurement",
                "target",
                "lsl",
                "usl",
                "pred_prob_oos",
                "pred_prob_rule",
                "current_status",
                "pct_of_spec_used",
                "temperature",
                "pressure",
                "speed",
                "humidity",
                "wear_index",
                "is_oos_now",
            ]
            rename_map = {
                "part_display": "Part No",
                "machine_display": "Machine",
                "timestamp": "Latest Timestamp",
                "measurement": "Measurement",
                "target": "Target",
                "lsl": "LSL",
                "usl": "USL",
                "pred_prob_oos": "Future OOS Risk",
                "pred_prob_rule": "Future Rule Risk",
                "current_status": "Status",
                "pct_of_spec_used": "Spec Usage",
                "temperature": "Temperature",
                "pressure": "Pressure",
                "speed": "Speed",
                "humidity": "Humidity",
                "wear_index": "Wear Index",
                "is_oos_now": "Currently OOS",
            }

            display_df = status_df[show_cols].rename(columns=rename_map).copy()
            display_df["Future OOS Risk"] = display_df["Future OOS Risk"].map(lambda x: f"{x:.1%}")
            display_df["Future Rule Risk"] = display_df["Future Rule Risk"].map(lambda x: f"{x:.1%}")
            display_df["Spec Usage"] = display_df["Spec Usage"].map(lambda x: f"{x:.1%}")
            display_df["Currently OOS"] = display_df["Currently OOS"].map(
                lambda x: "Yes" if int(x) == 1 else "No"
            )

            st.plotly_chart(
                go.Figure(
                    go.Bar(
                        x=["Critical", "High", "Moderate", "Low"],
                        y=[
                            (status_df["current_status"] == "Critical").sum(),
                            (status_df["current_status"] == "High").sum(),
                            (status_df["current_status"] == "Moderate").sum(),
                            (status_df["current_status"] == "Low").sum(),
                        ],
                        name="Combos",
                    )
                ).update_layout(
                    title="Current Status Distribution",
                    xaxis_title="Status",
                    yaxis_title="Count",
                    height=360,
                ),
                use_container_width=True,
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# HOW IT WORKS
# -----------------------------------------------------------------------------
with st.expander("How this Predictive SPC Demo Works"):
    st.markdown(
        f"""
### What this app predicts
For each current sample, the app predicts whether in the **next {future_horizon} samples** there will be:
1. **Any out-of-spec event**
2. **Any Nelson-style rule event**

### Features used
The models learn from:
- Recent measurement behavior
- Rolling averages and standard deviations
- Slopes and moving range
- Distance to spec limits
- Recent instability history
- Process variables like temperature, pressure, speed, humidity, and wear
- Part, machine, and shift context

### Why this matters
Traditional SPC says **what already happened**.
Predictive SPC estimates **what is likely to happen next** so the process can be corrected earlier.

### Suggested next step for real deployment
Replace demo data with:
- SQL Server, historian, or OPC data
- real subgroup measurements
- actual Nelson rule engine
- cavity-aware logic
- plant-specific action rules
"""
    )