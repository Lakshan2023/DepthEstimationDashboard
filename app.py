import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "Videos")

st.set_page_config(
    page_title="Eigen vs DPT – Video Comparison",
    layout="wide"
)

st.title("Eigen Network vs DPT-Hybrid")
st.subheader("KITTI Dataset – Video Comparison")

st.divider()

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
