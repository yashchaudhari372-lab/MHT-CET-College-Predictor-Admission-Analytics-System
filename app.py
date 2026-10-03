# app.py
import json
import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Base Configuration & Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "historical_cutoffs.csv")
MODEL_PATH = os.path.join(BASE_DIR, "mhtcet_catboost.cbm")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.pkl")

app = Flask(__name__)

# Global singletons cached per instance
DATA_CACHE = None
MODEL_INSTANCE = None
METADATA_CACHE = None
VERIFIED_METRICS = {"test_rmse": 12.16, "train_rmse": 12.70}


def load_assets():
    """Safely and efficiently initializes data, metadata, and model artifacts."""
    global DATA_CACHE, MODEL_INSTANCE, METADATA_CACHE

    if DATA_CACHE is None:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Historical cutoff file missing: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        cat_cols = ["college", "branch", "category", "gender", "seat_type"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        num_cols = [
            "min_percentile",
            "cutoff_p10",
            "median_percentile",
            "max_percentile",
            "allotment_count",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "allotment_count" in df.columns:
            df["allotment_count"] = df["allotment_count"].astype(int)
        DATA_CACHE = df

    if METADATA_CACHE is None and os.path.exists(METADATA_PATH):
        try:
            import joblib

            METADATA_CACHE = joblib.load(METADATA_PATH)
        except Exception as e:
            app.logger.warning(f"Could not load bundled metadata: {e}")
            METADATA_CACHE = {
                "features": [
                    "college",
                    "branch",
                    "category",
                    "gender",
                    "seat_type",
                    "allotment_count",
                ],
                "categorical_features": [
                    "college",
                    "branch",
                    "category",
                    "gender",
                    "seat_type",
                ],
                "target": "cutoff_p10",
            }

    if MODEL_INSTANCE is None and os.path.exists(MODEL_PATH):
        try:
            from catboost import CatBoostRegressor

            cb = CatBoostRegressor()
            cb.load_model(MODEL_PATH)
            MODEL_INSTANCE = cb
        except Exception as e:
            app.logger.warning(f"CatBoost model bypassed: {e}")
            MODEL_INSTANCE = None


try:
    load_assets()
except Exception as err:
    print(f"Startup Warning: Asset loading error: {err}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/options", methods=["GET"])
def get_filter_options():
    if DATA_CACHE is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    categories = sorted(DATA_CACHE["category"].dropna().unique().tolist())
    branches = sorted(DATA_CACHE["branch"].dropna().unique().tolist())
    genders = sorted(DATA_CACHE["gender"].dropna().unique().tolist())
    seat_types = sorted(DATA_CACHE["seat_type"].dropna().unique().tolist())
    colleges = sorted(DATA_CACHE["college"].dropna().unique().tolist())

    return jsonify(
        {
            "categories": categories,
            "branches": branches,
            "genders": genders,
            "seat_types": seat_types,
            "colleges": colleges,
            "total_records": len(DATA_CACHE),
            "unique_colleges": len(colleges),
            "unique_branches": len(branches),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Malformed JSON payload"}), 400

    try:
        percentile = float(payload.get("percentile", -1))
        if not (0.0 <= percentile <= 100.0):
            return jsonify({"error": "Percentile must be between 0 and 100"}), 422
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid numerical value for percentile"}), 422

    category = str(payload.get("category", "")).strip()
    gender = str(payload.get("gender", "")).strip()
    branch = str(payload.get("branch", "")).strip()
    seat_type = str(payload.get("seat_type", "")).strip()
    college_filter = str(payload.get("college", "")).strip()
    limit = int(payload.get("limit", 30))
    margin = float(payload.get("margin", 3.0))

    if DATA_CACHE is None:
        return jsonify({"error": "Dataset currently unavailable"}), 500

    df_filtered = DATA_CACHE.copy()
    if category:
        df_filtered = df_filtered[df_filtered["category"] == category]
    if gender:
        df_filtered = df_filtered[df_filtered["gender"] == gender]
    if branch:
        df_filtered = df_filtered[df_filtered["branch"] == branch]
    if seat_type:
        df_filtered = df_filtered[df_filtered["seat_type"] == seat_type]
    if college_filter:
        df_filtered = df_filtered[
            df_filtered["college"].str.contains(college_filter, case=False, na=False)
        ]

    total_matched = len(df_filtered)
    if total_matched == 0:
        return jsonify(
            {
                "success": True,
                "count": 0,
                "total_matched": 0,
                "recommendations": [],
                "analytics": {},
                "message": "No colleges match your active criteria. Try broadening branch or quota selections.",
            }
        )

    df_filtered["diff"] = percentile - df_filtered["cutoff_p10"]
    df_sorted = df_filtered.sort_values(
        by=["diff", "allotment_count"], ascending=[False, False]
    )
    recommendations_slice = df_sorted.head(limit).copy()

    features_list = (
        METADATA_CACHE.get("features", [])
        if METADATA_CACHE
        else [
            "college",
            "branch",
            "category",
            "gender",
            "seat_type",
            "allotment_count",
        ]
    )

    predictions = []
    if (
        MODEL_INSTANCE is not None
        and len(recommendations_slice) > 0
        and set(features_list).issubset(recommendations_slice.columns)
    ):
        try:
            inf_pool = recommendations_slice[features_list].copy()
            for cat_col in ["college", "branch", "category", "gender", "seat_type"]:
                if cat_col in inf_pool.columns:
                    inf_pool[cat_col] = inf_pool[cat_col].astype(str)
            raw_preds = MODEL_INSTANCE.predict(inf_pool)
            predictions = np.clip(raw_preds, 0.0, 100.0).tolist()
        except Exception as e:
            app.logger.error(f"Inference error: {e}")
            predictions = [None] * len(recommendations_slice)
    else:
        predictions = [None] * len(recommendations_slice)

    results = []
    for idx, (_, row) in enumerate(recommendations_slice.iterrows()):
        hist_p10 = round(float(row["cutoff_p10"]), 4)
        pred_p10 = (
            round(float(predictions[idx]), 4)
            if idx < len(predictions) and predictions[idx] is not None
            else None
        )
        diff_val = round(percentile - hist_p10, 4)

        if percentile >= hist_p10:
            status_group = "Safe Zone"
            status_desc = "Your percentile meets or exceeds historical CAP Round I P10."
            badge_color = "success"
        elif abs(diff_val) <= margin:
            status_group = "Target / Reach"
            status_desc = f"Within competitive reach margin (±{margin}%)."
            badge_color = "warning"
        else:
            status_group = "Ambitious"
            status_desc = "Substantially above historical cutoff threshold."
            badge_color = "danger"

        results.append(
            {
                "college": row["college"],
                "branch": row["branch"],
                "category": row["category"],
                "gender": row["gender"],
                "seat_type": row["seat_type"],
                "allotment_count": int(row["allotment_count"]),
                "min_percentile": round(float(row["min_percentile"]), 2),
                "historical_cutoff_p10": hist_p10,
                "median_percentile": round(float(row["median_percentile"]), 2),
                "max_percentile": round(float(row["max_percentile"]), 2),
                "model_predicted_p10": pred_p10,
                "percentile_difference": diff_val,
                "group_status": status_group,
                "group_desc": status_desc,
                "badge_color": badge_color,
            }
        )

    hist_cutoffs = df_filtered["cutoff_p10"].values
    analytics = {
        "filtered_count": int(total_matched),
        "mean_cutoff": round(float(np.mean(hist_cutoffs)), 2) if len(hist_cutoffs) > 0 else 0,
        "min_cutoff": round(float(np.min(hist_cutoffs)), 2) if len(hist_cutoffs) > 0 else 0,
        "max_cutoff": round(float(np.max(hist_cutoffs)), 2) if len(hist_cutoffs) > 0 else 0,
        "verified_metrics": VERIFIED_METRICS,
        "sample_cutoffs": hist_cutoffs[:400].tolist(),
    }

    return jsonify(
        {
            "success": True,
            "count": len(results),
            "total_matched": total_matched,
            "student_percentile": percentile,
            "recommendations": results,
            "analytics": analytics,
        }
    )


@app.route("/api/download", methods=["POST"])
def download_csv():
    try:
        data = request.get_json(force=True)
        items = data.get("items", [])
        if not items:
            return Response("No records selected", status=400)

        df = pd.DataFrame(items)
        cols_to_export = [
            "college",
            "branch",
            "category",
            "seat_type",
            "historical_cutoff_p10",
            "model_predicted_p10",
            "percentile_difference",
            "group_status",
            "allotment_count",
        ]
        available_cols = [c for c in cols_to_export if c in df.columns]
        csv_buffer = df[available_cols].to_csv(index=False)

        return Response(
            csv_buffer,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment;filename=mhtcet_evaluated_cutoffs.csv"
            },
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Embedded Multicolor Dynamic SPA Frontend
# ---------------------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" data-theme="nordic">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CET Matrix | High-Precision Admission Engine</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://unpkg.com/feather-icons"></script>
  
  <style>
    /* -----------------------------------------------------------------------
       Multi-Color Dynamic Theme Variables
       ----------------------------------------------------------------------- */
    :root, [data-theme="nordic"] {
      --bg-base: #0b0f19;
      --bg-surface: #111827;
      --bg-card: rgba(17, 24, 39, 0.85);
      --bg-glass: rgba(11, 15, 25, 0.78);
      --border: rgba(56, 189, 248, 0.16);
      --border-focus: #38bdf8;
      --primary: #38bdf8;
      --primary-gradient: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
      --accent: #818cf8;
      --accent-glow: rgba(56, 189, 248, 0.28);
      --text-heading: #f8fafc;
      --text-body: #94a3b8;
      --tag-safe: #10b981;
      --tag-target: #f59e0b;
      --tag-ambitious: #f43f5e;
      --font-mono: 'JetBrains Mono', monospace;
    }

    [data-theme="cyber"] {
      --bg-base: #07090e;
      --bg-surface: #0f131f;
      --bg-card: rgba(15, 19, 31, 0.9);
      --bg-glass: rgba(7, 9, 14, 0.85);
      --border: rgba(236, 72, 153, 0.22);
      --border-focus: #ec4899;
      --primary: #ec4899;
      --primary-gradient: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
      --accent: #a855f7;
      --accent-glow: rgba(236, 72, 153, 0.35);
      --text-heading: #ffffff;
      --text-body: #cbd5e1;
      --tag-safe: #06b6d4;
      --tag-target: #eab308;
      --tag-ambitious: #f43f5e;
    }

    [data-theme="emerald"] {
      --bg-base: #061412;
      --bg-surface: #0b1f1c;
      --bg-card: rgba(11, 31, 28, 0.9);
      --bg-glass: rgba(6, 20, 18, 0.82);
      --border: rgba(52, 211, 153, 0.2);
      --border-focus: #10b981;
      --primary: #10b981;
      --primary-gradient: linear-gradient(135deg, #10b981 0%, #14b8a6 50%, #f59e0b 100%);
      --accent: #fbbf24;
      --accent-glow: rgba(16, 185, 129, 0.3);
      --text-heading: #ecfdf5;
      --text-body: #99f6e4;
      --tag-safe: #10b981;
      --tag-target: #f59e0b;
      --tag-ambitious: #ef4444;
    }

    [data-theme="crimson"] {
      --bg-base: #140b0f;
      --bg-surface: #1f1219;
      --bg-card: rgba(31, 18, 25, 0.9);
      --bg-glass: rgba(20, 11, 15, 0.85);
      --border: rgba(244, 63, 94, 0.22);
      --border-focus: #f43f5e;
      --primary: #f43f5e;
      --primary-gradient: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
      --accent: #fb7185;
      --accent-glow: rgba(244, 63, 94, 0.35);
      --text-heading: #fff1f2;
      --text-body: #fecdd3;
      --tag-safe: #10b981;
      --tag-target: #f59e0b;
      --tag-ambitious: #f43f5e;
    }

    /* -----------------------------------------------------------------------
       Reset & Foundation
       ----------------------------------------------------------------------- */
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      transition: background-color 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                  border-color 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                  color 0.2s ease;
    }

    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg-base);
      color: var(--text-body);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.55;
    }

    /* Ambient dynamic glow blobs */
    .ambient-glow-1 {
      position: fixed;
      top: -160px;
      right: -140px;
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, var(--accent-glow) 0%, transparent 68%);
      filter: blur(100px);
      z-index: 0;
      pointer-events: none;
    }

    .ambient-glow-2 {
      position: fixed;
      bottom: -160px;
      left: -140px;
      width: 650px;
      height: 650px;
      background: radial-gradient(circle, var(--accent-glow) 0%, transparent 68%);
      filter: blur(120px);
      z-index: 0;
      pointer-events: none;
    }

    /* Header Nav */
    .header-bar {
      position: sticky;
      top: 0;
      z-index: 80;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.95rem 2.25rem;
      background: var(--bg-glass);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
    }

    .brand-cluster {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
    }

    .brand-icon-box {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: var(--primary-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      box-shadow: 0 4px 14px var(--accent-glow);
    }

    .brand-title {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: var(--text-heading);
    }

    .brand-title span {
      background: var(--primary-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .theme-selector-box {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
    }

    .theme-chip {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      cursor: pointer;
      border: 2px solid transparent;
      outline: 2px solid transparent;
    }

    .theme-chip.active {
      outline-color: var(--text-heading);
    }

    .chip-nordic { background: #38bdf8; }
    .chip-cyber { background: #ec4899; }
    .chip-emerald { background: #10b981; }
    .chip-crimson { background: #f43f5e; }

    /* Layout Wrapper */
    .app-viewport {
      position: relative;
      z-index: 10;
      max-width: 1540px;
      margin: 0 auto;
      padding: 1.8rem 2.25rem 4rem;
    }

    /* Metrics Strip */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.1rem;
      margin-bottom: 2rem;
    }

    .kpi-tile {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1.15rem 1.35rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 20px -3px rgba(0, 0, 0, 0.35);
    }

    .kpi-tag {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-body);
      margin-bottom: 0.4rem;
    }

    .kpi-number {
      font-size: 1.9rem;
      font-weight: 800;
      color: var(--text-heading);
      letter-spacing: -0.03em;
    }

    /* Workspace 2-Column Split */
    .workspace-split {
      display: grid;
      grid-template-columns: 410px 1fr;
      gap: 1.85rem;
      align-items: start;
    }

    @media (max-width: 1100px) {
      .workspace-split {
        grid-template-columns: 1fr;
      }
    }

    /* Panels */
    .glass-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 1.7rem;
      backdrop-filter: blur(14px);
      box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
      margin-bottom: 1.85rem;
    }

    .box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.3rem;
    }

    .box-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 0.55rem;
    }

    /* Form Design */
    .field-row {
      margin-bottom: 1.05rem;
    }

    .field-label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-body);
      margin-bottom: 0.4rem;
    }

    .input-stylish {
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.75rem 1rem;
      color: var(--text-heading);
      font-size: 0.92rem;
      outline: none;
      font-family: inherit;
    }

    .input-stylish:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .btn-exec {
      width: 100%;
      border: none;
      padding: 0.95rem;
      border-radius: 0.8rem;
      font-size: 0.95rem;
      font-weight: 700;
      color: #fff;
      background: var(--primary-gradient);
      box-shadow: 0 4px 18px var(--accent-glow);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .btn-exec:hover {
      opacity: 0.94;
      transform: translateY(-1px);
    }

    .btn-outline {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-body);
      padding: 0.45rem 0.85rem;
      border-radius: 0.55rem;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }

    .btn-outline:hover {
      color: var(--text-heading);
      border-color: var(--border-focus);
    }

    /* Cards Feed */
    .cards-grid {
      display: grid;
      gap: 1.15rem;
    }

    .college-entry {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1.35rem;
      position: relative;
      overflow: hidden;
    }

    .college-entry:hover {
      border-color: var(--primary);
    }

    .entry-headline {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      margin-bottom: 0.4rem;
    }

    .college-heading {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-heading);
    }

    .branch-pill {
      display: inline-block;
      font-size: 0.82rem;
      color: var(--primary);
      font-weight: 600;
      margin-bottom: 0.85rem;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .badge-success { background: rgba(16, 185, 129, 0.15); color: var(--tag-safe); border: 1px solid var(--tag-safe); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: var(--tag-target); border: 1px solid var(--tag-target); }
    .badge-danger  { background: rgba(244, 63, 94, 0.15); color: var(--tag-ambitious); border: 1px solid var(--tag-ambitious); }

    .meta-tags-line {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 0.9rem;
      font-size: 0.78rem;
      color: var(--text-body);
    }

    .meta-tag {
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 0.2rem 0.55rem;
      border-radius: 0.4rem;
    }

    /* Stat Counters Grid */
    .metric-cells {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.75rem;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.8rem;
    }

    .metric-cell {
      display: flex;
      flex-direction: column;
    }

    .cell-title {
      font-size: 0.68rem;
      color: var(--text-body);
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
    }

    .cell-value {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text-heading);
      font-family: var(--font-mono);
    }

    /* Comparison Drawer */
    .compare-tray {
      position: fixed;
      bottom: 1.6rem;
      right: 2rem;
      background: var(--bg-surface);
      border: 1px solid var(--primary);
      box-shadow: 0 10px 40px -10px rgba(0,0,0,0.7);
      border-radius: 1rem;
      padding: 0.85rem 1.5rem;
      display: none;
      align-items: center;
      gap: 1.25rem;
      z-index: 95;
    }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(8px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }

    .modal-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      width: 100%;
      max-width: 900px;
      max-height: 85vh;
      overflow-y: auto;
      padding: 2rem;
      position: relative;
    }

    /* Tab Switchers */
    .tab-pills {
      display: flex;
      gap: 0.5rem;
      background: var(--bg-surface);
      padding: 0.3rem;
      border-radius: 0.65rem;
      border: 1px solid var(--border);
    }

    .tab-pill {
      background: transparent;
      border: none;
      padding: 0.4rem 0.85rem;
      color: var(--text-body);
      font-size: 0.82rem;
      font-weight: 600;
      border-radius: 0.5rem;
      cursor: pointer;
    }

    .tab-pill.active {
      background: var(--primary-gradient);
      color: #fff;
    }

    .chart-panel {
      display: none;
    }

    .chart-panel.active {
      display: block;
    }

    .spinner {
      border: 3px solid rgba(255,255,255,0.15);
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border-left-color: #fff;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="ambient-glow-1"></div>
  <div class="ambient-glow-2"></div>

  <!-- Header Bar -->
  <header class="header-bar">
    <a href="#" class="brand-cluster">
      <div class="brand-icon-box">
        <i data-feather="crosshair"></i>
      </div>
      <div>
        <h1 class="brand-title">CET <span>Matrix</span></h1>
      </div>
    </a>

    <!-- Multicolor Switcher -->
    <div style="display: flex; align-items: center; gap: 1.25rem;">
      <div class="theme-selector-box" title="Select Multi-color Interface Theme">
        <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--text-body); margin-right: 0.2rem;">Theme:</span>
        <div class="theme-chip chip-nordic active" onclick="setTheme('nordic')" title="Nordic Cyan"></div>
        <div class="theme-chip chip-cyber" onclick="setTheme('cyber')" title="Cyberpunk Fuchsia"></div>
        <div class="theme-chip chip-emerald" onclick="setTheme('emerald')" title="Emerald Gold"></div>
        <div class="theme-chip chip-crimson" onclick="setTheme('crimson')" title="Crimson Titanium"></div>
      </div>
    </div>
  </header>

  <div class="app-viewport">
    <!-- Stat Strip -->
    <div class="kpi-strip">
      <div class="kpi-tile">
        <span class="kpi-tag">Official Quotas Indexed</span>
        <span class="kpi-number" id="kpi-records">56,833</span>
      </div>
      <div class="kpi-tile">
        <span class="kpi-tag">Participating Institutes</span>
        <span class="kpi-number" id="kpi-colleges">384</span>
      </div>
      <div class="kpi-tile">
        <span class="kpi-tag">Engineering Specializations</span>
        <span class="kpi-number" id="kpi-branches">142</span>
      </div>
      <div class="kpi-tile">
        <span class="kpi-tag">Evaluator Baseline RMSE</span>
        <span class="kpi-number" style="color: var(--primary);">12.16</span>
      </div>
    </div>

    <!-- Main Workspace Layout -->
    <div class="workspace-split">
      <!-- Input Sidebar -->
      <aside>
        <div class="glass-box">
          <div class="box-header">
            <h2 class="box-title"><i data-feather="filter"></i> Admission Factors</h2>
            <button class="btn-outline" onclick="resetForm()">Reset</button>
          </div>

          <form id="prediction-form" onsubmit="handlePredict(event)">
            <div class="field-row">
              <label class="field-label" for="pct-input">Your Percentile (0.00 – 100.00)</label>
              <input type="number" step="0.0001" min="0" max="100" class="input-stylish" id="pct-input" required placeholder="e.g. 96.8400">
            </div>

            <div class="field-row">
              <label class="field-label" for="category-select">Seat Category</label>
              <select class="input-stylish" id="category-select" required>
                <option value="">Fetching categories...</option>
              </select>
            </div>

            <div class="field-row">
              <label class="field-label" for="gender-select">Candidate Gender</label>
              <select class="input-stylish" id="gender-select" required>
                <option value="">Fetching genders...</option>
              </select>
            </div>

            <div class="field-row">
              <label class="field-label" for="branch-select">Target Specialization</label>
              <select class="input-stylish" id="branch-select">
                <option value="">All Engineering Branches</option>
              </select>
            </div>

            <div class="field-row">
              <label class="field-label" for="seat-type-select">Seat Quota Category</label>
              <select class="input-stylish" id="seat-type-select">
                <option value="">All Seat Types</option>
              </select>
            </div>

            <div class="field-row">
              <label class="field-label" for="college-input">Institute Filter (Optional)</label>
              <input type="text" class="input-stylish" id="college-input" placeholder="e.g. COEP, VJTI, SPIT, Pune">
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1.3rem;">
              <div>
                <label class="field-label" for="limit-select">Max Returns</label>
                <select class="input-stylish" id="limit-select">
                  <option value="15">Top 15</option>
                  <option value="30" selected>Top 30</option>
                  <option value="50">Top 50</option>
                  <option value="100">Top 100</option>
                </select>
              </div>
              <div>
                <label class="field-label" for="margin-input">Reach Band (±%)</label>
                <input type="number" step="0.5" min="0.5" max="10" class="input-stylish" id="margin-input" value="3.0">
              </div>
            </div>

            <button type="submit" class="btn-exec" id="submit-btn">
              <span>Execute Evaluation</span>
              <i data-feather="arrow-right"></i>
            </button>
          </form>
        </div>
      </aside>

      <!-- Results & Visualization Column -->
      <main>
        <!-- Analytical Visualizer -->
        <div class="glass-box">
          <div class="box-header">
            <h2 class="box-title"><i data-feather="pie-chart"></i> Distribution Dynamics</h2>
            <div class="tab-pills">
              <button class="tab-pill active" onclick="switchVisualTab('panel-dist', this)">Histogram</button>
              <button class="tab-pill" onclick="switchVisualTab('panel-scatter', this)">Actual vs Predicted</button>
            </div>
          </div>
          
          <div id="panel-dist" class="chart-panel active">
            <div id="chart-distribution" style="width: 100%; height: 320px;"></div>
          </div>
          <div id="panel-scatter" class="chart-panel">
            <div id="chart-scatter" style="width: 100%; height: 320px;"></div>
          </div>
        </div>

        <!-- College Recommendation Feed -->
        <div class="glass-box">
          <div class="box-header">
            <div>
              <h2 class="box-title"><i data-feather="check-square"></i> Matched Admission Paths</h2>
              <span id="results-count-text" style="font-size: 0.82rem; color: var(--text-body);">Enter your percentile in the sidebar to run analysis.</span>
            </div>
            <button class="btn-outline" id="export-btn" style="display:none;" onclick="exportResultsCSV()">
              <i data-feather="download"></i> Download CSV
            </button>
          </div>

          <div id="results-list" class="cards-grid">
            <div style="text-align: center; padding: 3rem 1rem; color: var(--text-body);">
              <i data-feather="shield" style="width: 44px; height: 44px; opacity: 0.3; margin-bottom: 0.8rem;"></i>
              <p>Configure candidate percentile and criteria in the sidebar to generate verified historical comparisons.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <!-- Selection Compare Tray -->
  <div class="compare-tray" id="compare-bar">
    <span id="compare-count-text" style="font-size: 0.88rem; font-weight: 600; color: var(--text-heading);">0 colleges selected</span>
    <button class="btn-exec" style="padding: 0.45rem 1rem; width: auto; font-size: 0.82rem;" onclick="viewComparisonModal()">Compare Now</button>
    <button class="btn-outline" style="padding: 0.45rem 0.85rem;" onclick="clearComparison()">Clear</button>
  </div>

  <!-- Comparison Modal -->
  <div class="modal-backdrop" id="compare-modal" onclick="closeModalOnBg(event)">
    <div class="modal-box">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-heading);">Side-by-Side Comparison</h3>
        <button class="btn-outline" onclick="closeComparisonModal()"><i data-feather="x"></i></button>
      </div>
      <div id="modal-table-container"></div>
    </div>
  </div>

  <script>
    feather.replace();
    let currentResults = [];
    let selectedColleges = new Map();

    // Multi-Color Theme Switcher
    function setTheme(themeName) {
      document.documentElement.setAttribute('data-theme', themeName);
      localStorage.setItem('cet_matrix_theme', themeName);
      
      document.querySelectorAll('.theme-chip').forEach(chip => chip.classList.remove('active'));
      const activeChip = document.querySelector(`.chip-${themeName}`);
      if (activeChip) activeChip.classList.add('active');

      refreshPlotsTheme();
    }

    // Startup Theme Retrieval
    const savedTheme = localStorage.getItem('cet_matrix_theme') || 'nordic';
    setTheme(savedTheme);

    async function loadOptions() {
      try {
        const res = await fetch('/api/options');
        const data = await res.json();
        
        populateSelect('category-select', data.categories, false);
        populateSelect('gender-select', data.genders, false);
        populateSelect('branch-select', data.branches, true, 'All Engineering Branches');
        populateSelect('seat-type-select', data.seat_types, true, 'All Seat Types');

        document.getElementById('kpi-records').innerText = data.total_records.toLocaleString();
        document.getElementById('kpi-colleges').innerText = data.unique_colleges.toLocaleString();
        document.getElementById('kpi-branches').innerText = data.unique_branches.toLocaleString();
      } catch (err) {
        console.error('Could not initialize options:', err);
      }
    }

    function populateSelect(id, list, addDefaultAll = false, defaultText = 'All') {
      const select = document.getElementById(id);
      select.innerHTML = '';
      if (addDefaultAll) {
        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.innerText = defaultText;
        select.appendChild(defaultOpt);
      }
      list.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.innerText = item;
        select.appendChild(opt);
      });
    }

    function switchVisualTab(tabId, btn) {
      document.querySelectorAll('.tab-pill').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.chart-panel').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    async function handlePredict(e) {
      e.preventDefault();
      const submitBtn = document.getElementById('submit-btn');
      const originalText = submitBtn.innerHTML;
      submitBtn.innerHTML = '<div class="spinner"></div> Calculating...';
      submitBtn.disabled = true;

      const payload = {
        percentile: parseFloat(document.getElementById('pct-input').value),
        category: document.getElementById('category-select').value,
        gender: document.getElementById('gender-select').value,
        branch: document.getElementById('branch-select').value,
        seat_type: document.getElementById('seat-type-select').value,
        college: document.getElementById('college-input').value,
        limit: parseInt(document.getElementById('limit-select').value),
        margin: parseFloat(document.getElementById('margin-input').value)
      };

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (data.success) {
          currentResults = data.recommendations;
          renderResults(data);
          renderAnalyticsPlots(data);
        } else {
          alert(data.error || 'Evaluation error occurred.');
        }
      } catch (err) {
        alert('Server or network failure.');
      } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        feather.replace();
      }
    }

    function renderResults(data) {
      const container = document.getElementById('results-list');
      const countLabel = document.getElementById('results-count-text');
      const exportBtn = document.getElementById('export-btn');

      if (!data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 2.5rem; color: var(--text-body);">
            <p>${data.message || 'No colleges matched criteria.'}</p>
          </div>
        `;
        countLabel.innerText = '0 matched colleges';
        exportBtn.style.display = 'none';
        return;
      }

      countLabel.innerText = `Showing top ${data.recommendations.length} of ${data.total_matched} matches`;
      exportBtn.style.display = 'inline-flex';

      container.innerHTML = data.recommendations.map((item, idx) => {
        const isChecked = selectedColleges.has(idx);
        return `
          <div class="college-entry">
            <div class="entry-headline">
              <div>
                <div class="college-heading">${item.college}</div>
                <div class="branch-pill">${item.branch}</div>
              </div>
              <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span class="status-badge badge-${item.badge_color}">${item.group_status}</span>
                <input type="checkbox" onchange="toggleSelect(${idx})" ${isChecked ? 'checked' : ''} style="cursor: pointer; transform: scale(1.2);">
              </div>
            </div>

            <div class="meta-tags-line">
              <span class="meta-tag">Quota: <strong>${item.seat_type}</strong></span>
              <span class="meta-tag">Category: <strong>${item.category}</strong></span>
              <span class="meta-tag">Gender: <strong>${item.gender}</strong></span>
              <span class="meta-tag">Allotments: <strong>${item.allotment_count}</strong></span>
            </div>

            <div class="metric-cells">
              <div class="metric-cell">
                <span class="cell-title">Observed Cutoff P10</span>
                <span class="cell-value">${item.historical_cutoff_p10.toFixed(2)}</span>
              </div>
              <div class="metric-cell">
                <span class="cell-title">Model Estimate P10</span>
                <span class="cell-value" style="color: var(--primary);">${item.model_predicted_p10 !== null ? item.model_predicted_p10.toFixed(2) : 'N/A'}</span>
              </div>
              <div class="metric-cell">
                <span class="cell-title">Percentile Margin</span>
                <span class="cell-value" style="color: ${item.percentile_difference >= 0 ? 'var(--tag-safe)' : 'var(--tag-ambitious)'};">
                  ${item.percentile_difference >= 0 ? '+' : ''}${item.percentile_difference.toFixed(2)}
                </span>
              </div>
              <div class="metric-cell">
                <span class="cell-title">Cohort Range</span>
                <span class="cell-value" style="font-size: 0.88rem;">${item.min_percentile.toFixed(1)} – ${item.max_percentile.toFixed(1)}</span>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function toggleSelect(index) {
      if (selectedColleges.has(index)) {
        selectedColleges.delete(index);
      } else {
        selectedColleges.set(index, currentResults[index]);
      }
      updateCompareBar();
    }

    function updateCompareBar() {
      const tray = document.getElementById('compare-bar');
      const countLabel = document.getElementById('compare-count-text');
      const count = selectedColleges.size;
      countLabel.innerText = `${count} option${count === 1 ? '' : 's'} selected`;
      tray.style.display = count > 0 ? 'flex' : 'none';
    }

    function clearComparison() {
      selectedColleges.clear();
      updateCompareBar();
      if (currentResults.length > 0) renderResults({ recommendations: currentResults, total_matched: currentResults.length });
    }

    function viewComparisonModal() {
      const modal = document.getElementById('compare-modal');
      const container = document.getElementById('modal-table-container');
      const items = Array.from(selectedColleges.values());

      if (items.length === 0) return;

      container.innerHTML = `
        <div style="overflow-x: auto;">
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem;">
            <thead>
              <tr style="border-bottom: 2px solid var(--border); color: var(--text-heading);">
                <th style="padding: 0.75rem;">College</th>
                <th style="padding: 0.75rem;">Branch</th>
                <th style="padding: 0.75rem;">Quota</th>
                <th style="padding: 0.75rem;">Hist P10</th>
                <th style="padding: 0.75rem;">Model P10</th>
                <th style="padding: 0.75rem;">Margin</th>
              </tr>
            </thead>
            <tbody>
              ${items.map(item => `
                <tr style="border-bottom: 1px solid var(--border);">
                  <td style="padding: 0.75rem; font-weight: 600; color: var(--text-heading);">${item.college}</td>
                  <td style="padding: 0.75rem;">${item.branch}</td>
                  <td style="padding: 0.75rem;">${item.seat_type}</td>
                  <td style="padding: 0.75rem; font-family: var(--font-mono);">${item.historical_cutoff_p10.toFixed(2)}</td>
                  <td style="padding: 0.75rem; font-family: var(--font-mono); color: var(--primary);">${item.model_predicted_p10 !== null ? item.model_predicted_p10.toFixed(2) : 'N/A'}</td>
                  <td style="padding: 0.75rem; font-family: var(--font-mono); color: ${item.percentile_difference >= 0 ? 'var(--tag-safe)' : 'var(--tag-ambitious)'};">
                    ${item.percentile_difference >= 0 ? '+' : ''}${item.percentile_difference.toFixed(2)}
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
      modal.style.display = 'flex';
      feather.replace();
    }

    function closeComparisonModal() {
      document.getElementById('compare-modal').style.display = 'none';
    }

    function closeModalOnBg(e) {
      if (e.target.id === 'compare-modal') closeComparisonModal();
    }

    function renderAnalyticsPlots(data) {
      const computedStyles = getComputedStyle(document.documentElement);
      const textColor = computedStyles.getPropertyValue('--text-heading').trim() || '#f8fafc';
      const primaryColor = computedStyles.getPropertyValue('--primary').trim() || '#38bdf8';
      const gridColor = 'rgba(255,255,255,0.06)';

      const cutoffs = data.analytics.sample_cutoffs || [];
      const studentScore = data.student_percentile;

      const distTrace = {
        x: cutoffs,
        type: 'histogram',
        nbinsx: 24,
        marker: { color: primaryColor, opacity: 0.75 },
        name: 'Historical Cutoffs'
      };

      const distLayout = {
        title: { text: 'Distribution of Historical Cutoffs', font: { color: textColor, size: 13 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textColor, family: 'Plus Jakarta Sans' },
        margin: { t: 40, b: 40, l: 40, r: 20 },
        shapes: [{
          type: 'line',
          x0: studentScore,
          x1: studentScore,
          y0: 0,
          y1: 1,
          yref: 'paper',
          line: { color: '#fbbf24', width: 2, dash: 'dash' }
        }],
        annotations: [{
          x: studentScore,
          y: 0.95,
          yref: 'paper',
          text: `Your Percentile (${studentScore.toFixed(2)})`,
          showarrow: true,
          arrowcolor: '#fbbf24',
          font: { color: '#fbbf24', size: 10 }
        }],
        xaxis: { gridcolor: gridColor, title: 'Percentile' },
        yaxis: { gridcolor: gridColor, title: 'Quotas' }
      };

      Plotly.newPlot('chart-distribution', [distTrace], distLayout, { responsive: true, displayModeBar: false });

      const obs = [];
      const preds = [];
      const labels = [];
      data.recommendations.forEach(r => {
        if (r.model_predicted_p10 !== null) {
          obs.push(r.historical_cutoff_p10);
          preds.push(r.model_predicted_p10);
          labels.push(r.college);
        }
      });

      const scatterTrace = {
        x: obs,
        y: preds,
        text: labels,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 8, color: primaryColor, opacity: 0.8 },
        name: 'Quota Node'
      };

      const fitLine = {
        x: [0, 100],
        y: [0, 100],
        mode: 'lines',
        line: { color: '#94a3b8', dash: 'dot' },
        name: 'Ideal Fit'
      };

      const scatterLayout = {
        title: { text: 'Actual vs Estimated Historical P10', font: { color: textColor, size: 13 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textColor, family: 'Plus Jakarta Sans' },
        margin: { t: 40, b: 40, l: 40, r: 20 },
        xaxis: { gridcolor: gridColor, title: 'Historical Cutoff P10', range: [50, 100] },
        yaxis: { gridcolor: gridColor, title: 'Model Estimated P10', range: [50, 100] },
        showlegend: false
      };

      Plotly.newPlot('chart-scatter', [scatterTrace, fitLine], scatterLayout, { responsive: true, displayModeBar: false });
    }

    function refreshPlotsTheme() {
      if (currentResults.length > 0) {
        Plotly.relayout('chart-distribution', {});
        Plotly.relayout('chart-scatter', {});
      }
    }

    async function exportResultsCSV() {
      if (!currentResults || currentResults.length === 0) return;
      const res = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: currentResults })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'mhtcet_evaluated_cutoffs.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    function resetForm() {
      document.getElementById('prediction-form').reset();
      currentResults = [];
      clearComparison();
      document.getElementById('results-list').innerHTML = `
        <div style="text-align: center; padding: 3rem 1rem; color: var(--text-body);">
          <p>Criteria reset. Input your percentile to recalculate.</p>
        </div>
      `;
      document.getElementById('results-count-text').innerText = 'Criteria reset.';
      document.getElementById('export-btn').style.display = 'none';
    }

    window.addEventListener('DOMContentLoaded', () => {
      loadOptions();
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
