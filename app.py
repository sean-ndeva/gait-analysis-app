import streamlit as st

st.set_page_config(page_title="Gait Analysis App", layout="centered")
st.title("👣 Gait Analysis App")

st.write("Welcome! Upload a short walking video (MP4), and we'll analyze knee angles.")

video_file = st.file_uploader("Upload a walking video", type=["mp4"])

if video_file is not None:
    st.video(video_file)
    st.success("✅ Video uploaded successfully. (Analysis coming soon...)")


