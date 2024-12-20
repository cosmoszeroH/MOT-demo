import streamlit as st
import tempfile

from tracking import tracking_object_in_frames

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    tracking_object_in_frames(temp_path, '')