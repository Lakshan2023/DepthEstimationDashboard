import os
import streamlit as st
import pandas as pd

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
    st.markdown("## Eigen Network (2014)")
    st.video(
        os.path.join(VIDEO_DIR, "eigen_gray.mp4"),
        width=VIDEO_WIDTH
    )

with col2:
    st.markdown("## DPT-Hybrid (Transformer)")
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
    ],
    "Eigen et al. (Coarse + Fine)": [
        0.710,
        0.897,
        0.964,
        0.195,
        1.417,
        6.056,
        0.262,
    ],
    "DPT-Hybrid (Current Implementation)": [
        0.959,
        0.995,
        0.999,
        0.062,
        0.222,
        2.573,
        0.092,
    ],
}

df = pd.DataFrame(data)

# Metrics where LOWER is better
lower_better = {"Abs Rel", "Sq Rel", "RMSE", "RMSE log"}

def highlight_best(row):
    vals = row[1:]
    best = vals.min() if row["Metric"] in lower_better else vals.max()
    return [
        "font-weight: bold; background-color: #d4f4dd"
        if v == best else ""
        for v in row
    ]

styled_df = (
    df.style
    .apply(highlight_best, axis=1)
    .format("{:.3f}", subset=df.columns[1:])
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Table: Quantitative comparison of Eigen Network (2014) and DPT-Hybrid "
    "on the KITTI Eigen split. Values are reported to three decimal places. "
    "Green-highlighted values indicate the best performance per metric."
)
