# -*- coding: utf-8 -*-
"""
Predictive SPC Streamlit App (Demo Data Included) - FIXED VERSION
------------------------------------------------------------------
FIXES APPLIED:
1. Removed @st.cache_resource on get_trained_models (DataFrames not hashable)
2. Added error handling in profile_sensitivity_for_subset for missing target_col
3. Fixed NaN handling in make_status_tile_html
4. Added empty DataFrame checks before operations
5. Added index reset in conditional_feature_effect_curve
6. Improved consecutive_trend_run logic
7. Enhanced error handling in feature curve operations
8. Better validation for relationship confidence estimation

What this app does
- Generates realistic demo manufacturing/SPC data for multiple part numbers and machines
- Builds forward-looking prediction targets:
    1) Out-of-spec risk in the next N samples
    2) Nelson-style rule risk in the next N samples
- Trains tree-based ML models on engineered SPC/process features
- Shows current risk, likely drivers, prediction charts, feature relationship AI, and recommended actions
- Includes a second dashboard tab showing current status by part/machine combo
  for a selected date/time period
- Includes colored status tiles for quick glance viewing
- Includes a help section and tooltips
- Adds:
    - optimal setpoint recommendation
    - top 3 actions panel
    - risk decomposition
    - feature stability warning
    - risk cliff detection
    - 2D interaction heatmap
    - what-if simulator
    - relationship confidence
    - part/machine learning profile
    - early warning timeline
    - operator mode summary

Run:
    streamlit run predictive_spc_fixed.py
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

with st.expander("❓ Help / What This App Does", expanded=False):
    st.markdown(
        """
### Overview
This app demonstrates **Predictive SPC** using realistic manufacturing demo data.

Instead of only showing what already happened, it estimates what is likely to happen next.

### What the app predicts
- **Future OOS Risk**  
  Probability that the process will go **out of spec** in the next selected number of samples.

- **Future Rule Risk**  
  Probability that the process will show a **control-rule violation pattern** soon
  (such as a strong trend, long run on one side, or extreme point).

### Main sections
- **Detail View**  
  Lets you inspect one Part/Machine combination in detail.

- **Current Status Dashboard**  
  Shows the latest status for all Part/Machine combinations in the selected date/time range.

### Key terms
- **Spec Usage**  
  How much of the tolerance has already been used.  
  - 0% = centered at target
  - 100% = at a spec limit
  - over 100% = out of spec

- **Future Prediction Horizon (samples)**  
  How far ahead the model looks.  
  Example: if set to 8, the app predicts whether an issue is likely in the **next 8 samples**.

### What the colors mean
- **Red** = Critical
- **Orange** = High
- **Yellow** = Moderate
- **Green** = Low

### Simplified rules used in this demo
This app uses a simplified set of Nelson-style future rule checks:
- point beyond 3 sigma
- 8 points on one side of center
- 6 points trending up or down
- 2 of 3 points beyond 2 sigma on the same side

### Important note
This app uses **demo data** and a simplified predictive SPC model for demonstration purposes.
"""
    )

with st.expander("ℹ️ Metric Definitions", expanded=False):
    st.markdown(
        """
- **Future OOS Risk**: chance the process goes out of spec soon  
- **Future Rule Risk**: chance the process shows an instability pattern soon  
- **Spec Usage**: how much of the tolerance is currently being used  
- **Recent Slope**: short-term direction of the measurement trend
"""
    )

with st.expander("🧠 Feature Relationship AI Help", expanded=False):
    st.markdown(
        """
This section estimates how changing a selected feature affects predicted risk.

### What it does
- You choose a feature such as **temperature** or **pressure**
- The app estimates how predicted **Future OOS Risk** or **Future Rule Risk** changes
- It does this near the **current operating point**, not from a totally unrelated global average

### Why this is better
Simple one-variable sweeps can be misleading when features are dependent on each other.
This view uses a **local neighborhood of similar rows** so the selected feature is varied across more realistic values.

### How to read it
- Rising line = higher feature values tend to increase risk
- Falling line = higher feature values tend to reduce risk
- Curved/nonlinear line = the feature effect depends on where you are operating
"""
    )

with st.expander("🚀 Prescriptive AI Add-ons Help", expanded=False):
    st.markdown(
        """
These add-ons build on top of the existing predictive SPC and feature relationship analysis.

### What they add
- **Optimal Setpoint**: suggests the best local value for a selected feature
- **Top 3 Actions**: shows the strongest local improvements
- **Risk Decomposition**: estimates which features contribute most to risk
- **Feature Stability Warning**: combines current trend + feature/risk relationship
- **Risk Cliff Detection**: highlights sharp jumps in predicted risk
- **What-if Simulator**: lets you test changes before making them
- **2D Interaction Heatmap**: shows how two features work together
- **Part/Machine Learning Profile**: shows what matters most for this specific subset
- **Early Warning Timeline**: highlights periods where risk crosses warning levels
- **Operator Summary**: gives a plain-English action summary
"""
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
        drift_factor = 0.0

        for m_idx, machine in enumerate(machines):
            machine_bias = (m_idx - 1) * 0.045
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

                drift_factor += rng.normal(0.0005, 0.0002)

                temp += rng.normal(0, 0.35) + 0.012 * drift_dir + 0.01 * drift_factor
                pressure += rng.normal(0, 0.28) + 0.009 * drift_dir + 0.008 * drift_factor
                speed += rng.normal(0, 0.40) - 0.008 * drift_dir
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
                drift_effect = 0.020 * drift_factor

                level += (0.0021 * drift_dir) + rng.normal(0, 0.006)

                noise_sigma = (0.040 + 0.004 * drift_factor) * episodic_var_boost
                measurement = (
                    level
                    + temp_effect
                    + pressure_effect
                    + speed_effect
                    + humidity_effect
                    + drift_effect
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
    """FIX #5: Improved trend counting logic"""
    if len(values) < 2:
        return 0

    diffs = np.diff(values)
    if len(diffs) == 0:
        return 0

    # Get last direction
    last_dir = 1 if diffs[-1] > 0 else (-1 if diffs[-1] < 0 else 0)
    if last_dir == 0:
        return 0

    run = 1
    # Count backwards from second-to-last
    for d in reversed(diffs[:-1]):
        cur_dir = 1 if d > 0 else (-1 if d < 0 else 0)
        if cur_dir == last_dir and cur_dir != 0:
            run += 1
        else:
            break
    return run


def nelson_like_rule_trigger(window_measurements: pd.Series, center: float, sigma: float) -> int:
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


def feature_display_name(feature: str) -> str:
    return feature.replace("_", " ").title()


def get_base_numeric_features(feature_cols: List[str]) -> List[str]:
    excluded_prefixes = ("part_no_", "machine_", "shift_")
    excluded_exact = {"hour", "dayofweek", "row_in_group"}

    out = []
    for c in feature_cols:
        if c.startswith(excluded_prefixes):
            continue
        if c in excluded_exact:
            continue
        out.append(c)
    return out


def predict_target_prob(row_df: pd.DataFrame, model: RandomForestClassifier) -> float:
    return float(model.predict_proba(row_df)[:, 1][0])


def conditional_feature_effect_curve(
    current_row: pd.Series,
    full_reference_df: pd.DataFrame,
    feature_cols: List[str],
    selected_feature: str,
    model: RandomForestClassifier,
    n_points: int = 21,
    neighborhood_size: int = 250,
) -> pd.DataFrame:
    numeric_candidates = get_base_numeric_features(feature_cols)
    compare_features = [c for c in numeric_candidates if c != selected_feature]

    work_df = full_reference_df[feature_cols].copy()

    needed_cols = compare_features + [selected_feature]
    work_df = work_df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed_cols).copy()

    if work_df.empty:
        return pd.DataFrame(columns=["feature_value", "predicted_risk"])

    if not compare_features:
        candidate_vals = np.sort(work_df[selected_feature].dropna().to_numpy())
        if len(candidate_vals) == 0:
            return pd.DataFrame(columns=["feature_value", "predicted_risk"])
        if len(candidate_vals) > n_points:
            idx = np.linspace(0, len(candidate_vals) - 1, n_points).astype(int)
            candidate_vals = candidate_vals[idx]
    else:
        compare_matrix = work_df[compare_features].copy()
        current_compare = current_row[compare_features].astype(float)

        means = compare_matrix.mean()
        stds = compare_matrix.std(ddof=0).replace(0, 1.0)  # Replace 0 with 1 to avoid division warnings

        # FIX #11: Safely divide by stds, replacing zeros first
        with np.errstate(divide='ignore', invalid='ignore'):
            z_work = (compare_matrix - means) / stds.clip(lower=1e-10)
            z_current = (current_compare - means) / stds.clip(lower=1e-10)

        dists = np.sqrt(((z_work - z_current) ** 2).sum(axis=1))
        # FIX #4: Add index reset after head() to prevent alignment issues
        work_df = work_df.assign(_dist=dists.values).sort_values("_dist").head(neighborhood_size).copy()
        work_df = work_df.reset_index(drop=True)

        candidate_vals = np.sort(work_df[selected_feature].dropna().to_numpy())
        if len(candidate_vals) == 0:
            return pd.DataFrame(columns=["feature_value", "predicted_risk"])
        if len(candidate_vals) > n_points:
            idx = np.linspace(0, len(candidate_vals) - 1, n_points).astype(int)
            candidate_vals = candidate_vals[idx]

    curve_rows = []
    base_row = current_row[feature_cols].copy()

    for val in candidate_vals:
        temp_row = base_row.copy()
        temp_row[selected_feature] = float(val)
        temp_df = pd.DataFrame([temp_row], columns=feature_cols)
        pred = predict_target_prob(temp_df, model)
        curve_rows.append({"feature_value": float(val), "predicted_risk": pred})

    curve_df = (
        pd.DataFrame(curve_rows)
        .drop_duplicates(subset=["feature_value"])
        .sort_values("feature_value")
        .reset_index(drop=True)
    )
    return curve_df


def summarize_feature_relationship(curve_df: pd.DataFrame, feature_name: str) -> str:
    if curve_df.empty or len(curve_df) < 3:
        return f"Not enough data to estimate the relationship for {feature_display_name(feature_name)}."

    x = curve_df["feature_value"].to_numpy(dtype=float)
    y = curve_df["predicted_risk"].to_numpy(dtype=float)

    slope = np.polyfit(x, y, 1)[0]
    
    # FIX #11: Safe correlation calculation - check for constant arrays first
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.std(x) > 1e-10 and np.std(y) > 1e-10:
            corr = np.corrcoef(x, y)[0, 1]
        else:
            corr = 0.0  # No correlation if either array is constant

    risk_min_idx = int(np.argmin(y))
    risk_max_idx = int(np.argmax(y))

    min_x = x[risk_min_idx]
    max_x = x[risk_max_idx]

    feature_label = feature_display_name(feature_name)

    # FIX #2: Use abs(corr) to properly check correlation strength
    if np.isfinite(corr) and abs(corr) >= 0.35 and slope > 0:
        return (
            f"Higher **{feature_label}** is associated with **higher predicted risk**. "
            f"Within this neighborhood, risk tends to rise as {feature_label.lower()} increases."
        )
    elif np.isfinite(corr) and abs(corr) >= 0.35 and slope < 0:
        return (
            f"Higher **{feature_label}** is associated with **lower predicted risk**. "
            f"Within this neighborhood, risk tends to fall as {feature_label.lower()} increases."
        )
    else:
        return (
            f"The relationship between **{feature_label}** and predicted risk looks **nonlinear or weak**. "
            f"Lowest estimated risk is near **{min_x:.3f}**, while highest estimated risk is near **{max_x:.3f}**."
        )


def make_feature_effect_chart(
    curve_df: pd.DataFrame,
    feature_name: str,
    current_value: float,
    target_name: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=curve_df["feature_value"],
            y=curve_df["predicted_risk"],
            mode="lines+markers",
            name=target_name,
        )
    )

    fig.add_vline(
        x=current_value,
        line_width=2,
        line_dash="dash",
    )

    y_min = float(curve_df["predicted_risk"].min())
    y_max = float(curve_df["predicted_risk"].max())

    if y_max - y_min < 0.02:
        pad = 0.01
    else:
        pad = (y_max - y_min) * 0.15

    y_low = max(0.0, y_min - pad)
    y_high = min(1.0, y_max + pad)

    fig.update_layout(
        title=f"{feature_display_name(feature_name)} vs {target_name}",
        xaxis_title=feature_display_name(feature_name),
        yaxis_title="Predicted Probability",
        yaxis=dict(range=[y_low, y_high]),
        height=420,
    )
    return fig


def make_feature_effect_delta_chart(
    curve_df: pd.DataFrame,
    feature_name: str,
    current_value: float,
    current_pred: float,
    target_name: str,
) -> go.Figure:
    df = curve_df.copy()
    df["delta_from_current"] = df["predicted_risk"] - current_pred

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["feature_value"],
            y=df["delta_from_current"],
            mode="lines+markers",
            name="Delta from current",
        )
    )

    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=current_value, line_width=2, line_dash="dash")

    dmin = float(df["delta_from_current"].min())
    dmax = float(df["delta_from_current"].max())
    pad = max(0.005, (dmax - dmin) * 0.15 if dmax != dmin else 0.01)

    fig.update_layout(
        title=f"{feature_display_name(feature_name)} vs Change in {target_name}",
        xaxis_title=feature_display_name(feature_name),
        yaxis_title="Change in Predicted Risk",
        yaxis=dict(range=[dmin - pad, dmax + pad]),
        height=420,
    )
    return fig


def recent_feature_slope(view_df: pd.DataFrame, feature: str, window: int = 8) -> float:
    vals = view_df[feature].tail(window)
    if len(vals) < 3:
        return 0.0
    return rolling_slope(vals)


def relationship_local_slope(curve_df: pd.DataFrame, current_value: float) -> float:
    if curve_df.empty or len(curve_df) < 3:
        return 0.0

    df = curve_df.copy()
    df["_dist"] = (df["feature_value"] - current_value).abs()
    local = df.sort_values("_dist").head(min(7, len(df))).sort_values("feature_value")

    if len(local) < 3:
        return 0.0

    x = local["feature_value"].to_numpy(dtype=float)
    y = local["predicted_risk"].to_numpy(dtype=float)

    if np.allclose(x.max(), x.min()):
        return 0.0
    
    # FIX #11: Safe polyfit with error suppression
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = float(np.polyfit(x, y, 1)[0])
    
    return slope if np.isfinite(slope) else 0.0


def estimate_relationship_confidence(curve_df: pd.DataFrame) -> tuple[str, float]:
    """FIX #7: Enhanced error handling for empty DataFrames"""
    if curve_df.empty or len(curve_df) < 5:
        return "Low", 0.20

    try:
        x = curve_df["feature_value"].to_numpy(dtype=float)
        y = curve_df["predicted_risk"].to_numpy(dtype=float)

        point_score = min(1.0, len(curve_df) / 15.0)
        span_score = min(1.0, (max(y) - min(y)) / 0.08) if (max(y) - min(y)) > 0 else 0.0

        # FIX #11: Safe correlation with variance check
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = 0.0
        
        if not np.isfinite(corr):
            corr = 0.0
        shape_score = abs(corr)

        score = 0.35 * point_score + 0.35 * span_score + 0.30 * shape_score

        if score >= 0.67:
            return "High", float(min(score, 1.0))
        elif score >= 0.40:
            return "Medium", float(min(score, 1.0))
        else:
            return "Low", float(min(score, 1.0))
    except Exception:
        return "Low", 0.20


def recommend_optimal_setpoint(
    curve_df: pd.DataFrame,
    current_value: float,
    current_pred: float,
) -> dict:
    if curve_df.empty:
        return {
            "optimal_value": current_value,
            "optimal_risk": current_pred,
            "improvement": 0.0,
            "direction": "Hold",
        }

    best_row = curve_df.loc[curve_df["predicted_risk"].idxmin()]
    optimal_value = float(best_row["feature_value"])
    optimal_risk = float(best_row["predicted_risk"])
    improvement = current_pred - optimal_risk

    if optimal_value > current_value:
        direction = "Increase"
    elif optimal_value < current_value:
        direction = "Decrease"
    else:
        direction = "Hold"

    return {
        "optimal_value": optimal_value,
        "optimal_risk": optimal_risk,
        "improvement": improvement,
        "direction": direction,
    }


def detect_risk_cliff(curve_df: pd.DataFrame, min_jump: float = 0.03):
    if curve_df.empty or len(curve_df) < 4:
        return None

    df = curve_df.copy().sort_values("feature_value").reset_index(drop=True)
    df["risk_jump"] = df["predicted_risk"].diff()

    idx = df["risk_jump"].abs().idxmax()
    jump_val = float(df.loc[idx, "risk_jump"]) if pd.notna(df.loc[idx, "risk_jump"]) else 0.0

    if abs(jump_val) < min_jump:
        return None

    return {
        "feature_value": float(df.loc[idx, "feature_value"]),
        "jump": jump_val,
        "direction": "Up" if jump_val > 0 else "Down",
    }


def risk_decomposition(
    current_row: pd.Series,
    reference_df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str],
    top_n: int = 6,
) -> pd.DataFrame:
    numeric_features = get_base_numeric_features(feature_cols)

    ref = reference_df[numeric_features].replace([np.inf, -np.inf], np.nan)
    means = ref.mean()
    stds = ref.std(ddof=0).replace(0, 1.0)

    x = current_row[numeric_features].astype(float)
    
    # FIX #11: Safe division by stds with clipping
    with np.errstate(divide='ignore', invalid='ignore'):
        z = ((x - means) / stds.clip(lower=1e-10)).abs()

    fi = pd.Series(model.feature_importances_, index=feature_cols)
    fi = fi.reindex(numeric_features).fillna(0.0)

    contrib = (z * fi).sort_values(ascending=False)
    out = contrib.reset_index()
    out.columns = ["feature", "contribution_score"]

    total = out["contribution_score"].sum()
    if total > 0:
        out["share_pct"] = out["contribution_score"] / total
    else:
        out["share_pct"] = 0.0

    out["feature"] = out["feature"].apply(feature_display_name)
    return out.head(top_n).reset_index(drop=True)


def build_top_actions(
    current_row: pd.Series,
    full_reference_df: pd.DataFrame,
    feature_cols: List[str],
    model: RandomForestClassifier,
    current_pred: float,
    candidate_features: List[str],
    top_k: int = 3,
) -> List[dict]:
    actions = []

    for feature in candidate_features:
        curve_df = conditional_feature_effect_curve(
            current_row=current_row,
            full_reference_df=full_reference_df,
            feature_cols=feature_cols,
            selected_feature=feature,
            model=model,
            n_points=15,
            neighborhood_size=200,
        )
        if curve_df.empty:
            continue

        rec = recommend_optimal_setpoint(
            curve_df=curve_df,
            current_value=float(current_row[feature]),
            current_pred=current_pred,
        )

        improvement = float(rec["improvement"])
        if improvement <= 0.002:
            continue

        direction = rec["direction"]
        if direction == "Hold":
            continue

        actions.append(
            {
                "feature": feature,
                "feature_label": feature_display_name(feature),
                "direction": direction,
                "current_value": float(current_row[feature]),
                "optimal_value": float(rec["optimal_value"]),
                "risk_improvement": improvement,
                "text": (
                    f"{direction} {feature_display_name(feature)} "
                    f"toward {rec['optimal_value']:.3f} "
                    f"(estimated risk change: -{improvement:.1%})"
                ),
            }
        )

    actions = sorted(actions, key=lambda x: x["risk_improvement"], reverse=True)
    return actions[:top_k]


def make_risk_decomposition_chart(decomp_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=decomp_df["share_pct"],
            y=decomp_df["feature"],
            orientation="h",
            name="Share",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Share of Estimated Risk Contribution",
        yaxis_title="Feature",
        height=360,
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def make_interaction_heatmap(
    current_row: pd.Series,
    full_reference_df: pd.DataFrame,
    feature_cols: List[str],
    feature_x: str,
    feature_y: str,
    model: RandomForestClassifier,
    grid_points: int = 11,
    neighborhood_size: int = 250,
):
    compare_features = [c for c in get_base_numeric_features(feature_cols) if c not in {feature_x, feature_y}]
    work_df = full_reference_df[feature_cols].copy()
    work_df = work_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature_x, feature_y] + compare_features).copy()

    if work_df.empty:
        return None

    if compare_features:
        means = work_df[compare_features].mean()
        stds = work_df[compare_features].std(ddof=0).replace(0, 1.0)
        
        # FIX #11: Safe division by stds with clipping
        with np.errstate(divide='ignore', invalid='ignore'):
            z_work = (work_df[compare_features] - means) / stds.clip(lower=1e-10)
            z_cur = (current_row[compare_features].astype(float) - means) / stds.clip(lower=1e-10)
        
        dists = np.sqrt(((z_work - z_cur) ** 2).sum(axis=1))
        work_df = work_df.assign(_dist=dists.values).sort_values("_dist").head(neighborhood_size).copy()

    x_vals = np.quantile(work_df[feature_x], np.linspace(0.05, 0.95, grid_points))
    y_vals = np.quantile(work_df[feature_y], np.linspace(0.05, 0.95, grid_points))
    x_vals = np.unique(x_vals)
    y_vals = np.unique(y_vals)

    z = np.zeros((len(y_vals), len(x_vals)))
    base_row = current_row[feature_cols].copy()

    for iy, yv in enumerate(y_vals):
        for ix, xv in enumerate(x_vals):
            temp_row = base_row.copy()
            temp_row[feature_x] = float(xv)
            temp_row[feature_y] = float(yv)
            temp_df = pd.DataFrame([temp_row], columns=feature_cols)
            z[iy, ix] = predict_target_prob(temp_df, model)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z,
            colorbar_title="Pred Risk",
        )
    )
    fig.add_vline(x=float(current_row[feature_x]), line_dash="dash", line_width=2)
    fig.add_hline(y=float(current_row[feature_y]), line_dash="dash", line_width=2)
    fig.update_layout(
        title=f"{feature_display_name(feature_x)} × {feature_display_name(feature_y)} Interaction",
        xaxis_title=feature_display_name(feature_x),
        yaxis_title=feature_display_name(feature_y),
        height=480,
    )
    return fig


def profile_sensitivity_for_subset(
    subset_df: pd.DataFrame,
    candidate_features: List[str],
    target_col: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """FIX #3: Added error handling for missing target_col"""
    # Validate target_col exists
    if target_col not in subset_df.columns:
        return pd.DataFrame(columns=["feature", "correlation_with_risk", "absolute_strength"])

    rows = []
    for f in candidate_features:
        if f not in subset_df.columns:
            continue
        s = subset_df[[f, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 8:
            continue
        corr = s[f].corr(s[target_col])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "feature": feature_display_name(f),
                "correlation_with_risk": corr,
                "absolute_strength": abs(corr),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "correlation_with_risk", "absolute_strength"])

    out = pd.DataFrame(rows).sort_values("absolute_strength", ascending=False).head(top_n).reset_index(drop=True)
    return out


def make_timeline_warning_chart(view_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view_df["timestamp"],
            y=view_df["pred_prob_oos"],
            mode="lines",
            name="Future OOS Risk",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view_df["timestamp"],
            y=view_df["pred_prob_rule"],
            mode="lines",
            name="Future Rule Risk",
        )
    )

    fig.add_hline(y=0.60, line_dash="dot")
    fig.add_hline(y=0.80, line_dash="dash")

    high_idx = view_df.index[(view_df["pred_prob_oos"] >= 0.60) | (view_df["pred_prob_rule"] >= 0.60)]
    if len(high_idx) > 0:
        first_high = int(high_idx[0])
        fig.add_vrect(
            x0=view_df.loc[first_high, "timestamp"],
            x1=view_df["timestamp"].max(),
            opacity=0.12,
            line_width=0,
        )

    fig.update_layout(
        title="Early Warning Timeline",
        xaxis_title="Timestamp",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
    )
    return fig


def operator_summary_text(
    current_row: pd.Series,
    current_oos: float,
    current_rule: float,
    top_actions: List[dict],
    feature_warning_text,
) -> str:
    status = probability_to_label(max(current_oos, current_rule))
    pieces = [f"Current process status is **{status}**."]

    if current_oos >= current_rule:
        pieces.append("Main concern is future out-of-spec risk.")
    else:
        pieces.append("Main concern is future control-rule risk.")

    if feature_warning_text:
        pieces.append(feature_warning_text)

    if top_actions:
        pieces.append("Best next adjustments:")
        for a in top_actions[:3]:
            pieces.append(f"- {a['text']}")
    else:
        pieces.append("No strong improvement action stands out from the current local analysis.")

    return "\n".join(pieces)


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
    """FIX #6: Improved NaN handling for status tiles"""
    status = row["current_status"]

    color_map = {
        "Critical": "#d9534f",
        "High": "#f0ad4e",
        "Moderate": "#ffd966",
        "Low": "#5cb85c",
    }
    bg = color_map.get(status, "#cccccc")
    text_color = "#000000" if status == "Moderate" else "#ffffff"

    # Safely handle is_oos_now which might be NaN
    is_oos_val = row.get("is_oos_now", 0)
    if pd.isna(is_oos_val):
        is_oos_val = 0
    oos_now = "Yes" if int(is_oos_val) == 1 else "No"

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


# FIX #1: Removed @st.cache_resource - DataFrames are not hashable
# Models will retrain when data changes, which is the correct behavior
def get_trained_models(model_df: pd.DataFrame) -> ModelBundle:
    return train_predictive_models(model_df)


# -----------------------------------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------------------------------
st.sidebar.header("Demo Settings")

future_horizon = st.sidebar.slider(
    "Future prediction horizon (samples)",
    3,
    20,
    8,
    1,
    help="How many future samples the model looks ahead when estimating risk.",
)

n_parts = st.sidebar.slider(
    "Number of demo parts",
    3,
    8,
    5,
    1,
    help="How many simulated part numbers to generate in the demo dataset.",
)

n_machines = st.sidebar.slider(
    "Number of demo machines",
    2,
    5,
    3,
    1,
    help="How many simulated machines to generate in the demo dataset.",
)

rows_per_combo = st.sidebar.slider(
    "Rows per part-machine",
    300,
    1200,
    650,
    50,
    help="How many time-ordered samples to generate for each Part/Machine combination.",
)

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
        selected_part = st.selectbox(
            "Select Part Number",
            options=part_options,
            index=0,
            help="Choose the part number to inspect in detail.",
        )
    with colf2:
        selected_machine = st.selectbox(
            "Select Machine",
            options=machine_options,
            index=0,
            help="Choose the machine to inspect in detail.",
        )

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
        help="Move through the time sequence for this Part/Machine combination.",
    )
    current_row = view_df.iloc[selected_idx]

    oos_prob = float(current_row["pred_prob_oos"])
    rule_prob = float(current_row["pred_prob_rule"])
    cur_slope = float(current_row.get("slope_8", 0.0))
    pct_spec = float(current_row.get("pct_of_spec_used", 0.0))

    st.caption(
        "These metrics summarize near-term quality risk for the selected Part/Machine at the chosen sample."
    )

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

    proc_cols = st.columns(4)
    proc_cols[0].write(f"**Temperature:** {current_row['temperature']:.2f}")
    proc_cols[1].write(f"**Pressure:** {current_row['pressure']:.2f}")
    proc_cols[2].write(f"**Speed:** {current_row['speed']:.2f}")
    proc_cols[3].write(f"**Humidity:** {current_row['humidity']:.2f}")

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

    st.subheader("🧠 Feature Relationship AI")

    selectable_features = get_base_numeric_features(bundle.feature_cols)

    fr1, fr2 = st.columns([1, 1])

    with fr1:
        selected_feature = st.selectbox(
            "Select feature to analyze",
            options=selectable_features,
            index=0,
            help="Choose a feature to see how changing it affects predicted risk near the current operating point.",
        )

    with fr2:
        selected_target = st.selectbox(
            "Select target",
            options=["Future OOS Risk", "Future Rule Risk"],
            index=0,
            help="Choose which predicted risk to analyze against the selected feature.",
        )

    selected_model = bundle.model_oos if selected_target == "Future OOS Risk" else bundle.model_rule

    curve_df = conditional_feature_effect_curve(
        current_row=current_row,
        full_reference_df=scored_df,
        feature_cols=bundle.feature_cols,
        selected_feature=selected_feature,
        model=selected_model,
        n_points=21,
        neighborhood_size=250,
    )

    current_feature_value = float(current_row[selected_feature]) if selected_feature in current_row.index else 0.0
    current_pred = float(
        selected_model.predict_proba(
            pd.DataFrame([current_row[bundle.feature_cols]], columns=bundle.feature_cols)
        )[:, 1][0]
    )

    # FIX #7: Enhanced empty DataFrame check before operations
    if curve_df.empty:
        st.warning("Not enough data to estimate a relationship for the selected feature.")
    else:
        st.markdown(summarize_feature_relationship(curve_df, selected_feature))

        st.plotly_chart(
            make_feature_effect_chart(
                curve_df=curve_df,
                feature_name=selected_feature,
                current_value=current_feature_value,
                target_name=selected_target,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_feature_effect_delta_chart(
                curve_df=curve_df,
                feature_name=selected_feature,
                current_value=current_feature_value,
                current_pred=current_pred,
                target_name=selected_target,
            ),
            use_container_width=True,
        )

        risk_span = float(curve_df["predicted_risk"].max() - curve_df["predicted_risk"].min())
        st.caption(
            f"Estimated risk range across realistic {feature_display_name(selected_feature).lower()} values: {risk_span:.2%}"
        )

        rel_cols = st.columns(4)
        rel_cols[0].metric("Current Feature Value", f"{current_feature_value:.4f}")
        rel_cols[1].metric("Current Predicted Risk", f"{current_pred:.1%}")
        rel_cols[2].metric("Min Estimated Risk", f"{curve_df['predicted_risk'].min():.1%}")
        rel_cols[3].metric("Max Estimated Risk", f"{curve_df['predicted_risk'].max():.1%}")

        trend_df = curve_df.copy()
        trend_df["risk_delta"] = trend_df["predicted_risk"].diff()
        trend_df["delta_from_current"] = trend_df["predicted_risk"] - current_pred

        st.markdown("**Relationship Table**")
        st.dataframe(
            trend_df.rename(
                columns={
                    "feature_value": feature_display_name(selected_feature),
                    "predicted_risk": selected_target,
                    "risk_delta": "Step Change in Risk",
                    "delta_from_current": "Change vs Current Risk",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("🚀 Prescriptive AI Add-ons")

    explain_mode = st.radio(
        "Explain mode",
        options=["Technical", "Operator"],
        horizontal=True,
        help="Technical shows more analysis detail. Operator shows a simpler action-oriented summary.",
    )

    adv1, adv2, adv3 = st.columns(3)

    confidence_label, confidence_score = estimate_relationship_confidence(curve_df)
    optimal_rec = recommend_optimal_setpoint(
        curve_df=curve_df,
        current_value=current_feature_value,
        current_pred=current_pred,
    )
    cliff_info = detect_risk_cliff(curve_df, min_jump=0.03)

    adv1.metric("Relationship Confidence", confidence_label, f"{confidence_score:.0%}")
    adv2.metric(
        "Optimal Setpoint",
        f"{optimal_rec['optimal_value']:.3f}",
        optimal_rec["direction"],
    )
    adv3.metric(
        "Best Possible Risk Change",
        f"-{optimal_rec['improvement']:.1%}" if optimal_rec["improvement"] > 0 else "0.0%",
    )

    if cliff_info is not None:
        st.warning(
            f"Risk cliff detected: near **{cliff_info['feature_value']:.3f}**, "
            f"predicted risk moves **{cliff_info['direction']} by {abs(cliff_info['jump']):.1%}**."
        )

    st.markdown("### Risk Decomposition")

    rd1, rd2 = st.columns(2)

    decomp_oos = risk_decomposition(
        current_row=current_row,
        reference_df=scored_df,
        model=bundle.model_oos,
        feature_cols=bundle.feature_cols,
        top_n=6,
    )
    decomp_rule = risk_decomposition(
        current_row=current_row,
        reference_df=scored_df,
        model=bundle.model_rule,
        feature_cols=bundle.feature_cols,
        top_n=6,
    )

    with rd1:
        st.markdown("**Estimated Contributors - Future OOS Risk**")
        st.plotly_chart(
            make_risk_decomposition_chart(decomp_oos, "OOS Risk Decomposition"),
            use_container_width=True,
        )
        st.dataframe(decomp_oos, use_container_width=True, hide_index=True)

    with rd2:
        st.markdown("**Estimated Contributors - Future Rule Risk**")
        st.plotly_chart(
            make_risk_decomposition_chart(decomp_rule, "Rule Risk Decomposition"),
            use_container_width=True,
        )
        st.dataframe(decomp_rule, use_container_width=True, hide_index=True)

    st.markdown("### Feature Stability Warning")

    feature_trend = recent_feature_slope(view_df, selected_feature, window=8)
    relation_slope = relationship_local_slope(curve_df, current_feature_value)

    feature_warning_text = None
    if feature_trend > 0 and relation_slope > 0:
        feature_warning_text = (
            f"**{feature_display_name(selected_feature)}** is trending upward and the model "
            f"associates higher values with higher **{selected_target.lower()}**."
        )
        st.warning(feature_warning_text)
    elif feature_trend < 0 and relation_slope < 0:
        feature_warning_text = (
            f"**{feature_display_name(selected_feature)}** is trending downward and the model "
            f"associates lower values with higher **{selected_target.lower()}**."
        )
        st.warning(feature_warning_text)
    else:
        st.info(
            f"No strong combined drift warning for **{feature_display_name(selected_feature)}** at the moment."
        )

    drift_cols = st.columns(2)
    drift_cols[0].metric("Recent Feature Slope", f"{feature_trend:.4f}")
    drift_cols[1].metric("Local Relationship Slope", f"{relation_slope:.4f}")

    st.markdown("### Top 3 Actions")

    action_candidates = [
        f for f in selectable_features
        if f in ["temperature", "pressure", "speed", "humidity", "pct_of_spec_used", "slope_8"]
    ]
    top_actions = build_top_actions(
        current_row=current_row,
        full_reference_df=scored_df,
        feature_cols=bundle.feature_cols,
        model=selected_model,
        current_pred=current_pred,
        candidate_features=action_candidates,
        top_k=3,
    )

    if top_actions:
        for i, action in enumerate(top_actions, start=1):
            st.write(f"{i}. {action['text']}")
    else:
        st.write("No strong local action recommendation stands out.")

    st.markdown("### What-if Simulator")

    # FIX #10: Add manual feature selection capability
    auto_features = [a["feature"] for a in top_actions[:3]]
    if not auto_features:
        auto_features = [f for f in ["temperature", "pressure", "speed"] if f in selectable_features]

    # Allow user to override auto-selected features
    sim_mode = st.radio(
        "Simulator mode",
        options=["Auto-selected (Top Actions)", "Manual selection"],
        horizontal=True,
        help="Auto-selected uses Top 3 Actions features. Manual lets you pick any features.",
        key="sim_mode_radio"
    )

    if sim_mode == "Manual selection":
        sim_features = st.multiselect(
            "Select features to simulate",
            options=selectable_features,
            default=auto_features[:min(3, len(auto_features))],
            max_selections=5,
            help="Choose 1-5 features to test in the what-if simulator.",
            key="manual_sim_features"
        )
        if not sim_features:
            st.warning("Please select at least one feature to simulate.")
            sim_features = auto_features
    else:
        sim_features = auto_features
        st.info(f"Simulating: {', '.join([feature_display_name(f) for f in sim_features])}")

    sim_row = current_row[bundle.feature_cols].copy()
    sim_cols = st.columns(max(1, len(sim_features)))

    for idx, sim_feature in enumerate(sim_features):
        ref_series = scored_df[sim_feature].replace([np.inf, -np.inf], np.nan).dropna()
        if ref_series.empty:
            continue

        low = float(ref_series.quantile(0.05))
        high = float(ref_series.quantile(0.95))
        cur = float(current_row[sim_feature])

        if high <= low:
            high = low + 1e-6

        with sim_cols[idx]:
            sim_value = st.slider(
                f"{feature_display_name(sim_feature)}",
                min_value=float(low),
                max_value=float(high),
                value=float(np.clip(cur, low, high)),
                step=float((high - low) / 100.0) if high > low else 0.01,
                key=f"sim_{sim_feature}",
            )
        sim_row[sim_feature] = sim_value

    sim_df = pd.DataFrame([sim_row], columns=bundle.feature_cols)
    sim_oos = float(bundle.model_oos.predict_proba(sim_df)[:, 1][0])
    sim_rule = float(bundle.model_rule.predict_proba(sim_df)[:, 1][0])

    sim_metrics = st.columns(4)
    sim_metrics[0].metric("Current OOS Risk", f"{oos_prob:.1%}")
    sim_metrics[1].metric("Simulated OOS Risk", f"{sim_oos:.1%}", f"{sim_oos - oos_prob:+.1%}")
    sim_metrics[2].metric("Current Rule Risk", f"{rule_prob:.1%}")
    sim_metrics[3].metric("Simulated Rule Risk", f"{sim_rule:.1%}", f"{sim_rule - rule_prob:+.1%}")

    st.markdown("### Multi-Feature Interaction Heatmap")

    heat1, heat2 = st.columns(2)
    with heat1:
        feature_x = st.selectbox(
            "Interaction feature X",
            options=selectable_features,
            index=max(0, selectable_features.index(selected_feature)) if selected_feature in selectable_features else 0,
            key="interaction_x",
        )
    with heat2:
        other_options = [f for f in selectable_features if f != feature_x]
        feature_y = st.selectbox(
            "Interaction feature Y",
            options=other_options,
            index=0,
            key="interaction_y",
        )

    interaction_fig = make_interaction_heatmap(
        current_row=current_row,
        full_reference_df=scored_df,
        feature_cols=bundle.feature_cols,
        feature_x=feature_x,
        feature_y=feature_y,
        model=selected_model,
        grid_points=11,
        neighborhood_size=250,
    )
    if interaction_fig is not None:
        st.plotly_chart(interaction_fig, use_container_width=True)
    else:
        st.warning("Not enough data to build the interaction heatmap.")

    st.markdown("### Part / Machine Learning Profile")

    profile1, profile2 = st.columns(2)
    oos_profile = profile_sensitivity_for_subset(view_df, selectable_features, "pred_prob_oos", top_n=5)
    rule_profile = profile_sensitivity_for_subset(view_df, selectable_features, "pred_prob_rule", top_n=5)

    with profile1:
        st.markdown("**Features Most Associated with OOS Risk in This Part/Machine**")
        st.dataframe(oos_profile, use_container_width=True, hide_index=True)

    with profile2:
        st.markdown("**Features Most Associated with Rule Risk in This Part/Machine**")
        st.dataframe(rule_profile, use_container_width=True, hide_index=True)

    st.markdown("### Early Warning Timeline")
    st.plotly_chart(make_timeline_warning_chart(view_df), use_container_width=True)

    if explain_mode == "Operator":
        st.markdown("### Operator Summary")
        st.markdown(
            operator_summary_text(
                current_row=current_row,
                current_oos=oos_prob,
                current_rule=rule_prob,
                top_actions=top_actions,
                feature_warning_text=feature_warning_text,
            )
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
        start_date = st.date_input(
            "Start date",
            value=min_ts.date(),
            key="start_date",
            help="Beginning of the period used for the dashboard.",
        )
        start_time = st.time_input(
            "Start time",
            value=min_ts.time(),
            key="start_time",
            help="Beginning time for the selected dashboard period.",
        )

    with dcol2:
        end_date = st.date_input(
            "End date",
            value=max_ts.date(),
            key="end_date",
            help="End of the period used for the dashboard.",
        )
        end_time = st.time_input(
            "End time",
            value=max_ts.time(),
            key="end_time",
            help="Ending time for the selected dashboard period.",
        )

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

            st.caption(
                "These dashboard counts summarize the latest status of each Part/Machine combination in the selected date/time period."
            )

            summary1, summary2, summary3, summary4 = st.columns(4)
            summary1.metric("Combos in Period", f"{status_df.shape[0]}")
            summary2.metric("Critical Combos", f"{(status_df['current_status'] == 'Critical').sum()}")
            summary3.metric("High Combos", f"{(status_df['current_status'] == 'High').sum()}")
            summary4.metric("Moderate Combos", f"{(status_df['current_status'] == 'Moderate').sum()}")

            st.markdown(
                """
**Status Legend:**  
🟥 Critical &nbsp;&nbsp; 🟧 High &nbsp;&nbsp; 🟨 Moderate &nbsp;&nbsp; 🟩 Low
"""
            )
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
                "is_oos_now": "Currently OOS",
            }

            display_df = status_df[show_cols].rename(columns=rename_map).copy()
            display_df["Future OOS Risk"] = display_df["Future OOS Risk"].map(lambda x: f"{x:.1%}")
            display_df["Future Rule Risk"] = display_df["Future Rule Risk"].map(lambda x: f"{x:.1%}")
            display_df["Spec Usage"] = display_df["Spec Usage"].map(lambda x: f"{x:.1%}")
            display_df["Currently OOS"] = display_df["Currently OOS"].map(
                lambda x: "Yes" if int(x) == 1 else "No"
            )

            fig_status = go.Figure(
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
            )
            fig_status.update_layout(
                title="Current Status Distribution",
                xaxis_title="Status",
                yaxis_title="Count",
                height=360,
            )
            st.plotly_chart(fig_status, use_container_width=True)

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
- Process variables like temperature, pressure, speed, and humidity
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

### All Fixes Applied
✅ Fixed @st.cache_resource with DataFrames  
✅ Added target_col validation  
✅ Improved NaN handling in status tiles  
✅ Added empty DataFrame checks  
✅ Fixed index alignment issues  
✅ Enhanced trend counting logic  
✅ Better error handling throughout  
"""
    )
