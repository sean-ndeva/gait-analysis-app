import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio


st.set_page_config(page_title="Gait Analysis", layout="centered")
st.title("👣 Gait Analysis App")
st.sidebar.header("Optional Personal Information")

height_cm = st.sidebar.slider("Your height (cm)", 140, 210, 175)
weight_kg = st.sidebar.slider("Your weight (kg)", 40, 150, 70)
age_group = st.sidebar.selectbox("Your age group", ["18-24", "25-30", "31-40", "41-50", "51-60", "60+"])

bmi = weight_kg / ((height_cm / 100) ** 2)
st.sidebar.write(f"**Your BMI:** {bmi:.1f}")

st.sidebar.header("Customize Detection Sensitivity")

asym_threshold = st.sidebar.slider(
    "Max acceptable angle difference (°)", 
    min_value=5, max_value=20, value=10
)

stiff_threshold = st.sidebar.slider(
    "Min acceptable knee flexion (°)", 
    min_value=90, max_value=140, value=120
)

# Display user inputs
st.markdown(f"""
**User Info**  
- Height: {height_cm} cm  
- Weight: {weight_kg} kg  
- Age group: {age_group}  
- BMI: {bmi:.1f}
""")

st.write("Upload a short walking video (MP4) and we'll analyze your knee angles.")

# 1️⃣ Upload video
video_file = st.file_uploader("Choose a walking video...", type=["mp4"])
if not video_file:
    st.stop()

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

with st.spinner("🔍 Analyzing video..."):
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        for frame in reader:
            results = pose.process(frame)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                def calc(a, b, c):
                    a, b, c = np.array(a), np.array(b), np.array(c)
                    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    deg = abs(np.degrees(rad))
                    return 360 - deg if deg > 180 else deg
                left = calc([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
                right = calc([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
                left_angles.append(left)
                right_angles.append(right)
        reader.close()
    except Exception as e:
        st.error(f"❌ Error reading video frames: {e}")
        st.stop()

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

# Download button
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download results as CSV", csv, "gait_analysis.csv", "text/csv")

# 7️⃣ Summary
mean_diff = df.Diff.mean()
max_diff  = df.Diff.max()

st.subheader("Summary")
st.write(f"- Mean left/right asymmetry: **{mean_diff:.1f}°**")
st.write(f"- Max asymmetry: **{max_diff:.1f}°**")
st.write(f"- Your BMI: **{bmi:.1f}**")

# Peer comparison
age_reference = {
    "18-24": 8,
    "25-30": 9,
    "31-40": 10,
    "41-50": 11,
    "51-60": 12,
    "60+": 14
}
normal_range = age_reference.get(age_group, 10)
st.write(f"- Expected average asymmetry for your age group ({age_group}): **{normal_range}°**")

if mean_diff > normal_range + 3:
    st.warning("⚠️ Your gait asymmetry is higher than typical for your age group. You may consider consulting a specialist.")
else:
    st.success("✅ Your gait asymmetry is within the normal range for your age group.")

# 8️⃣ Recommendations
st.subheader("Recommendations")
if mean_diff > asym_threshold:
    st.write("📌 **Possible Irregularity:** Gait asymmetry detected.")

    if mean_diff > normal_range + 5:
        st.error("🚨 The irregularity seems significant. Please consult a physiotherapist or orthopedic specialist.")

    st.write("🧘‍♀️ **Helpful Exercises:**")
    st.markdown("""
    - **Hip bridges**: Strengthen glutes and hamstrings  
    - **Leg swings**: Improve balance and flexibility  
    - **Step-ups**: Build lower body strength and coordination  
    - **Lunges with form control**: Improve symmetry  
    """)
else:
    st.write("🎉 No major issues detected. Keep up with regular walking or strength routines.")

