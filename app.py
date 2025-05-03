import streamlit as st
import tempfile, os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gait Analysis", layout="centered")
st.title("👣 Gait Analysis App")
st.write("Upload a short walking video (MP4) and we'll analyze your knee angles.")

# 1️⃣ Upload video
video_file = st.file_uploader("Choose a walking video...", type=["mp4"])
if not video_file:
    st.stop()  # wait until user uploads

# 2️⃣ Save to temp file
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(video_file.read())
video_path = tfile.name

# 3️⃣ Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 4️⃣ Process video and calculate angles
left_angles, right_angles = [], []
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def calc(a,b,c):
            a,b,c = np.array(a), np.array(b), np.array(c)
            rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            deg = abs(np.degrees(rad))
            return 360 - deg if deg>180 else deg
        left = calc([lm[23].x,lm[23].y],[lm[25].x,lm[25].y],[lm[27].x,lm[27].y])
        right= calc([lm[24].x,lm[24].y],[lm[26].x,lm[26].y],[lm[28].x,lm[28].y])
        left_angles.append(left)
        right_angles.append(right)
cap.release()
pose.close()
os.unlink(video_path)

# 5️⃣ Build DataFrame
df = pd.DataFrame({
    "Frame": range(len(left_angles)),
    "LeftKnee": left_angles,
    "RightKnee": right_angles
})
df["Diff"] = abs(df.LeftKnee - df.RightKnee)

# 6️⃣ Display results
st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Knee Angle Plot")
fig, ax = plt.subplots()
ax.plot(df.Frame, df.LeftKnee, label="Left Knee")
ax.plot(df.Frame, df.RightKnee, label="Right Knee")
ax.set_xlabel("Frame")
ax.set_ylabel("Angle (°)")
ax.legend()
st.pyplot(fig)

# 7️⃣ Summary
mean_diff = df.Diff.mean()
max_diff  = df.Diff.max()
bmi = None  # you can add height/weight inputs here if you like

st.subheader("Summary")
st.write(f"- Mean left/right asymmetry: **{mean_diff:.1f}°**")
st.write(f"- Max asymmetry: **{max_diff:.1f}°**")



