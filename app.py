import os
import streamlit as st
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "Videos")

st.set_page_config(
    page_title="Eigen vs DPT – Video Comparison",
    layout="wide"
)

st.title("Eigen Network vs DPT-Hybrid")
st.subheader("KITTI Dataset – Video Comparison")

st.divider()

# -------------------- Video Comparison --------------------
col1, col2 = st.columns([1, 1])

VIDEO_WIDTH = 640  # same width for both

with col1:
    st.markdown("## Eigen Network (2014) - Historical Approach")
    st.video(
        os.path.join(VIDEO_DIR, "eigen_gray.mp4"),
        width=VIDEO_WIDTH
    )

with col2:
    st.markdown("## DPT-Hybrid (Transformer) - Current Approach")
    st.video(
        os.path.join(VIDEO_DIR, "dpt.mp4"),
        width=VIDEO_WIDTH
    )

# -------------------- Benchmark Results --------------------
st.divider()
st.subheader("Benchmarking Results on KITTI Dataset")

import pandas as pd

data = {
    "Metric": [
        "δ < 1.25",
        "δ < 1.25²",
        "δ < 1.25³",
        "Abs Rel",
        "Sq Rel",
        "RMSE",
        "RMSE log",
        "SILog"
    ],
    "Eigen et al. (Coarse + Fine)": [
        0.710,
        0.897,
        0.964,
        0.195,
        1.417,
        6.056,
        0.262,
        0.240
    ],
    "DPT-Hybrid (Current Implementation)": [
        0.959,
        0.995,
        0.999,
        0.062,
        0.222,
        2.573,
        0.092,
        8.282
    ],
}

df = pd.DataFrame(data)

styled_df = (
    df.style
    .format("{:.3f}", subset=df.columns[1:])
    .set_table_styles([
        {"selector": "th", "props": [("color", "black"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("text-align", "center")]},
    ])
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Table: Quantitative comparison of Eigen Network (2014) and DPT-Hybrid "
    "on the KITTI Eigen split. Values are reported to three decimal places."
)


# -------------------- Virtual KITTI Benchmark --------------------
st.divider()
st.subheader("Virtual KITTI Benchmark – Challenging Conditions")

# =========================
# Eigen results
# =========================
conditions_eigen = {
    "clone":   [0.3043, 0.5806, 0.7547, 0.5085, 7.6954, 14.6812, 0.6832, 0.6736],
    "fog":     [0.2500, 0.4924, 0.6739, 0.5718, 8.4579, 15.9756, 0.7946, 0.7847],
    "morning": [0.3029, 0.5675, 0.7361, 0.5229, 7.5606, 14.3862, 0.6768, 0.6692],
    "overcast":[0.2939, 0.5605, 0.7241, 0.5321, 7.7903, 14.6906, 0.6900, 0.6823],
    "rain":    [0.2610, 0.5072, 0.6885, 0.5602, 8.2520, 15.6354, 0.7534, 0.7447],
    "sunset":  [0.2777, 0.5393, 0.7217, 0.5435, 7.8443, 14.7058, 0.6814, 0.6736],
}

# =========================
# DPT placeholders (LISTS)
# =========================
conditions_dpt = {
    "clone":   [0.850, 0.945, 0.972, 0.136, 0.758, 5.289, 0.224, 21.784],
    "fog":     [0.793, 0.908, 0.945, 0.157, 1.327, 7.207, 0.313, 29.944],
    "morning": [0.853, 0.944, 0.970, 0.134, 0.784, 5.410, 0.228, 22.149],
    "overcast":[0.828, 0.941, 0.969, 0.139, 0.803, 5.454, 0.231, 22.299],
    "rain":    [0.793, 0.904, 0.944, 0.160, 1.325, 7.104, 0.311, 29.635],
    "sunset":  [0.824, 0.937, 0.970, 0.140, 0.811, 5.467, 0.234, 22.665],
}

# =========================
# Metric names (COLUMNS)
# =========================
metrics = [
    "δ < 1.25",
    "δ < 1.25²",
    "δ < 1.25³",
    "Abs Rel",
    "Sq Rel",
    "RMSE",
    "RMSE log",
    "SIlog",
]

# =========================
# Build table rows
# =========================
rows = []

for condition in conditions_eigen.keys():
    # Eigen row
    rows.append({
        "Model": "Eigen Network (2014)",
        "Condition": condition,
        **dict(zip(metrics, conditions_eigen[condition]))
    })

    # DPT row (empty placeholders)
    rows.append({
        "Model": "DPT-Hybrid",
        "Condition": condition,
        **dict(zip(metrics, conditions_dpt[condition]))
    })

vk_df = pd.DataFrame(rows)

# =========================
# Style safely (numeric only)
# =========================
styled_vk_df = (
    vk_df.style
    .format("{:.3f}", subset=metrics, na_rep="—")
    .set_table_styles([
        {"selector": "th", "props": [("color", "black"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("text-align", "center")]},
        {"selector": "td.col0", "props": [("font-weight", "bold")]},
    ])
)

st.dataframe(
    styled_vk_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Table: Virtual KITTI benchmark results under challenging environmental conditions. "
    "For each condition, Eigen Network results are reported first, followed by "
    "DPT-Hybrid model results. (Each codition is tested with 2126 test samples)"
)
