import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2  # required for mediapipe even if not used directly
import imageio

# Streamlit page configuration
st.set_page_config(page_title="Gait Analysis", layout="centered")
st.title("👣 Stride Buddy App")

# Sidebar inputs
st.sidebar.header("Optional Personal Information")
height_cm = st.sidebar.slider("Your height (cm)", 140, 210, 175)
weight_kg = st.sidebar.slider("Your weight (kg)", 40, 150, 70)
age_group = st.sidebar.selectbox("Your age group", ["18-24", "25-30", "31-40", "41-50", "51-60", "60+"])

bmi = weight_kg / ((height_cm / 100) ** 2)
st.sidebar.write(f"**Your BMI:** {bmi:.1f}")

st.sidebar.header("Customize Detection Sensitivity")
asym_threshold = st.sidebar.slider("Max acceptable angle difference (°)", 5, 20, 10)
stiff_threshold = st.sidebar.slider("Min acceptable knee flexion (°)", 90, 140, 120)

# Display user info
st.markdown(f"""
**User Info**  
- Height: {height_cm} cm  
- Weight: {weight_kg} kg  
- Age group: {age_group}  
- BMI: {bmi:.1f}
""")

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Utility: extract all pose landmarks into an array
def extract_landmarks(video_path):
    reader = iio.imiter(video_path, plugin="pyav")
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    landmarks = []
    for frame in reader:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = pose.process(img)
        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            landmarks.append(row)
    pose.close()
    return np.array(landmarks)

# Utility: calculate asymmetry between two landmark indices
def calculate_asymmetry(landmarks, left_idx, right_idx):
    return np.abs(landmarks[:, left_idx*3] - landmarks[:, right_idx*3])

# Utility: display preview table of first/middle/last frames
def display_preview_table(landmarks):
    n = len(landmarks)
    frames = []
    if n >= 4:
        frames.extend(landmarks[:4])
        mid = n // 2
        frames.extend(landmarks[max(0, mid-2):mid+2])
        frames.extend(landmarks[-4:])
    df = pd.DataFrame(frames, columns=[f"LM{i+1}" for i in range(landmarks.shape[1])])
    st.subheader("Preview of Selected Frames")
    st.write(df)

# Utility: compute elbow angle (shoulder-elbow-wrist)
def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang

# Compute elbow asymmetry from landmarks array
def get_elbow_asymmetry(landmarks):
    left_angles, right_angles = [], []
    for frm in landmarks:
        ls = frm[11*3:11*3+3]; le = frm[13*3:13*3+3]; lw = frm[15*3:15*3+3]
        rs = frm[12*3:12*3+3]; re = frm[14*3:14*3+3]; rw = frm[16*3:16*3+3]
        left_angles.append(calculate_joint_angle(ls, le, lw))
        right_angles.append(calculate_joint_angle(rs, re, rw))
    return np.abs(np.array(left_angles) - np.array(right_angles))

# Main application logic
def main():
    st.markdown("### Upload Two Walking Videos (Opposite Directions)")
    col1, col2 = st.columns(2)
    with col1:
        vid1 = st.file_uploader("East-to-West", type=["mp4","mov","avi"], key="v1")
    with col2:
        vid2 = st.file_uploader("West-to-East", type=["mp4","mov","avi"], key="v2")

    # Process each uploaded video
    for label, video_file in [("East-to-West", vid1), ("West-to-East", vid2)]:
        if video_file:
            # Save temp
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(video_file.read()); path = tmp.name; tmp.close()
            st.subheader(f"Video Preview ({label})")
            st.video(video_file)

            # Preview frames
            reader = imageio.get_reader(path, "ffmpeg")
            cnt = reader.count_frames()
            for idx in [0, cnt//2, cnt-1]:
                st.image(reader.get_data(idx), caption=f"{label} Frame {idx}", use_column_width=True)
            reader.close()

            # Extract landmarks and show table
            landmarks = extract_landmarks(path)
            if landmarks.size == 0:
                st.error(f"No landmarks in {label} video.")
                os.remove(path)
                continue
            display_preview_table(landmarks)

            # Hip asymmetry
            hip_diff = calculate_asymmetry(landmarks, 23, 24)
            st.subheader(f"{label} Hip Asymmetry Over Time")
            fig, ax = plt.subplots()
            ax.plot(hip_diff, label="Hip X diff")
            ax.axhline(asym_threshold/100, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel("Frame"); ax.set_ylabel("Abs Diff")
            ax.legend(); st.pyplot(fig)
            st.write(f"Avg Hip Asymmetry ({label}): {np.mean(hip_diff):.4f}")

            # Arm asymmetry
            elbow_diff = get_elbow_asymmetry(landmarks)
            st.subheader(f"{label} Arm Swing Asymmetry Over Time")
            fig2, ax2 = plt.subplots()
            ax2.plot(elbow_diff, label="Elbow Diff")
            ax2.axhline(stiff_threshold/10, color='r', linestyle='--', label='Threshold')
            ax2.set_xlabel("Frame"); ax2.set_ylabel("Degrees")
            ax2.legend(); st.pyplot(fig2)
            st.write(f"Avg Arm Asymmetry ({label}): {np.mean(elbow_diff):.2f}°")

            os.remove(path)

if __name__ == "__main__":
    main()
