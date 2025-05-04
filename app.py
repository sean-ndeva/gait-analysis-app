import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2  # required for mediapipe even if not used directly
import imageio
from io import BytesIO

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

# Reference norms by age group
age_norms = {
    "18-24": {"hip": 0.02, "arm": 5},
    "25-30": {"hip": 0.025, "arm": 6},
    "31-40": {"hip": 0.03, "arm": 7},
    "41-50": {"hip": 0.035, "arm": 8},
    "51-60": {"hip": 0.04, "arm": 9},
    "60+": {"hip": 0.045, "arm": 10}
}
norms = age_norms.get(age_group, {"hip": 0.03, "arm": 7})

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_landmarks(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
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
    reader.close()
    pose.close()
    return np.array(landmarks)

def calculate_asymmetry(landmarks, left_idx, right_idx):
    return np.abs(landmarks[:, left_idx*3] - landmarks[:, right_idx*3])

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
    return df

def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang

def get_elbow_asymmetry(landmarks):
    left_angles, right_angles = [], []
    for frm in landmarks:
        ls = frm[11*3:11*3+3]; le = frm[13*3:13*3+3]; lw = frm[15*3:15*3+3]
        rs = frm[12*3:12*3+3]; re = frm[14*3:14*3+3]; rw = frm[16*3:16*3+3]
        left_angles.append(calculate_joint_angle(ls, le, lw))
        right_angles.append(calculate_joint_angle(rs, re, rw))
    return np.abs(np.array(left_angles) - np.array(right_angles))

def get_knee_flexion_stats(landmarks):
    left_angles, right_angles = [], []
    for frm in landmarks:
        lh = frm[23*3:23*3+3]; lk = frm[25*3:25*3+3]; la = frm[27*3:27*3+3]
        rh = frm[24*3:24*3+3]; rk = frm[26*3:26*3+3]; ra = frm[28*3:28*3+3]
        left_angles.append(calculate_joint_angle(lh, lk, la))
        right_angles.append(calculate_joint_angle(rh, rk, ra))
    left_min = np.min(left_angles)
    right_min = np.min(right_angles)
    return left_min, right_min, left_angles, right_angles

def main():
    st.markdown("### Upload Two Walking Videos (Opposite Directions)")
    col1, col2 = st.columns(2)
    with col1:
        vid1 = st.file_uploader("East-to-West", type=["mp4","mov","avi"], key="v1")
    with col2:
        vid2 = st.file_uploader("West-to-East", type=["mp4","mov","avi"], key="v2")

    comparisons = []

    for label, video_file in [("East-to-West", vid1), ("West-to-East", vid2)]:
        if video_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(video_file.read()); path = tmp.name; tmp.close()
            st.subheader(f"Video Preview ({label})")
            st.video(video_file)

            reader = imageio.get_reader(path, "ffmpeg")
            cnt = reader.count_frames()
            for idx in [0, cnt//2, cnt-1]:
                st.image(reader.get_data(idx), caption=f"{label} Frame {idx}", use_container_width=True)
            reader.close()

            landmarks = extract_landmarks(path)
            if landmarks.size == 0:
                st.error(f"No landmarks in {label} video.")
                os.remove(path)
                continue

            df_preview = display_preview_table(landmarks)
            hip_diff = calculate_asymmetry(landmarks, 23, 24)
            elbow_diff = get_elbow_asymmetry(landmarks)
            left_knee_min, right_knee_min, left_knees, right_knees = get_knee_flexion_stats(landmarks)

            avg_hip = np.mean(hip_diff)
            max_hip = np.max(hip_diff)
            avg_elbow = np.mean(elbow_diff)
            max_elbow = np.max(elbow_diff)
            comparisons.append((label, avg_hip, max_hip, avg_elbow, max_elbow, left_knee_min, right_knee_min))

            st.subheader(f"{label} Knee Flexion Range")
            fig, ax = plt.subplots()
            ax.plot(left_knees, label="Left Knee Angle")
            ax.plot(right_knees, label="Right Knee Angle")
            ax.axhline(stiff_threshold, color='r', linestyle='--', label='Stiffness Threshold')
            ax.set_xlabel("Frame"); ax.set_ylabel("Angle (°)")
            ax.legend(); st.pyplot(fig)
            st.write(f"Min Left Knee Flexion: {left_knee_min:.2f}°")
            st.write(f"Min Right Knee Flexion: {right_knee_min:.2f}°")

            st.subheader(f"{label} Hip Asymmetry Over Time")
            fig, ax = plt.subplots()
            ax.plot(hip_diff, label="Hip X diff")
            ax.axhline(asym_threshold/100, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel("Frame"); ax.set_ylabel("Abs Diff")
            ax.legend(); st.pyplot(fig)
            st.write(f"Avg Hip Asymmetry ({label}): {avg_hip:.4f}")
            st.write(f"Max Hip Asymmetry ({label}): {max_hip:.4f}")
            if avg_hip > norms["hip"]:
                st.warning(f"Hip asymmetry is higher than average for your age group ({norms['hip']:.3f}).")

            st.subheader(f"{label} Arm Swing Asymmetry Over Time")
            fig2, ax2 = plt.subplots()
            ax2.plot(elbow_diff, label="Elbow Diff")
            ax2.axhline(stiff_threshold/10, color='r', linestyle='--', label='Threshold')
            ax2.set_xlabel("Frame"); ax2.set_ylabel("Degrees")
            ax2.legend(); st.pyplot(fig2)
            st.write(f"Avg Arm Asymmetry ({label}): {avg_elbow:.2f}°")
            st.write(f"Max Arm Asymmetry ({label}): {max_elbow:.2f}°")
            if avg_elbow > norms["arm"]:
                st.warning(f"Arm swing asymmetry is higher than average for your age group ({norms['arm']}°).")

            if avg_elbow > stiff_threshold/10 or avg_hip > asym_threshold/100:
                st.subheader("Recommended Corrective Exercises")
                st.markdown("""
- **Hip bridges**: Strengthen glutes and hamstrings  
- **Leg swings**: Improve balance and flexibility  
- **Step-ups**: Build lower body strength and coordination  
- **Lunges with form control**: Improve symmetry  
""")

            csv_buffer = BytesIO()
            export_df = pd.DataFrame({
                "Hip_Diff": hip_diff,
                "Elbow_Diff": elbow_diff,
                "Left_Knee": left_knees,
                "Right_Knee": right_knees
            })
            export_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Asymmetry Data CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{label.replace(' ', '_')}_asymmetry_data.csv",
                mime="text/csv"
            )

            os.remove(path)

    if len(comparisons) == 2:
        st.subheader("Comparison Summary")
        df_compare = pd.DataFrame(comparisons, columns=["Video", "Avg Hip Asym", "Max Hip Asym", "Avg Arm Asym", "Max Arm Asym", "Min Left Knee", "Min Right Knee"])
        st.write(df_compare)

if __name__ == "__main__":
    main()
