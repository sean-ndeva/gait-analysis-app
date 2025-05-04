import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2  # Required for mediapipe even if not used directly

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_landmarks(video_path):
    video_reader = iio.imiter(video_path, plugin="pyav")
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    landmarks = []

    for frame in video_reader:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = pose.process(image)
        if results.pose_landmarks:
            landmark_row = []
            for lm in results.pose_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])
            landmarks.append(landmark_row)
    pose.close()
    return np.array(landmarks)

def calculate_asymmetry(landmarks, left_idx, right_idx):
    diffs = np.abs(landmarks[:, left_idx*3] - landmarks[:, right_idx*3])
    return diffs

def display_preview_table(landmarks):
    # Number of frames you want to display
    num_frames = len(landmarks)
    
    # Select frames: first 4, middle 4, and last 4 frames
    frames_to_show = []
    
    if num_frames >= 4:
        # First four frames
        frames_to_show.extend(landmarks[:4])
        
        # Middle four frames (or as many as possible if less than 4)
        middle_start = max(4, num_frames // 2 - 2)
        middle_end = min(num_frames, middle_start + 4)
        frames_to_show.extend(landmarks[middle_start:middle_end])
        
        # Last four frames
        frames_to_show.extend(landmarks[-4:])
    
    # Convert selected frames into a DataFrame for display in Streamlit
    df = pd.DataFrame(frames_to_show, columns=[f"Landmark {i+1}" for i in range(landmarks.shape[1])])
    
    # Show the table in Streamlit
    st.subheader("Preview of Selected Frames")
    st.write(df)

def main():
    st.title("Gait Analysis & Asymmetry Detection App")
    st.markdown("Upload a video for gait asymmetry analysis using MediaPipe.")
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.video(uploaded_file)

        with st.spinner("Analyzing video..."):
            landmarks = extract_landmarks(tmp_path)

        if landmarks.size == 0:
            st.error("No landmarks detected. Try a different video.")
            return

        # Display the preview table with selected frames
        display_preview_table(landmarks)

        # Calculate and display hip asymmetry (existing code)
        left_hip_idx, right_hip_idx = 23, 24
        hip_diff = calculate_asymmetry(landmarks, left_hip_idx, right_hip_idx)

        st.subheader("Hip Asymmetry Over Time")
        fig, ax = plt.subplots()
        ax.plot(hip_diff, label="Hip X-axis Difference")
        ax.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Asymmetry (abs diff)")
        ax.legend()
        st.pyplot(fig)

        avg_asymmetry = np.mean(hip_diff)
        st.write(f"Average Hip Asymmetry: **{avg_asymmetry:.4f}**")

        if avg_asymmetry < 0.05:
            st.success(f"Asymmetry of {avg_asymmetry:.4f} detected: within normal range, but improvements could help prevent future issues.")
        else:
            st.warning(f"Asymmetry of {avg_asymmetry:.4f} detected: outside normal range. Consider improving balance and gait.")

        os.remove(tmp_path)

if __name__ == "__main__":
    main()
