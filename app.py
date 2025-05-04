import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2  #required for mediapipe even if it is not used
import imageio

st.set_page_config(page_title="Gait Analysis", layout="centered")
st.title("👣 Stride Buddy App")
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

# 3️⃣ Display Video Preview
st.subheader("Video Preview (First, Middle, Last Frames)")

# Load the video and extract frames
reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = reader.count_frames()

# Extract first, middle, and last few frames
frames_to_preview = [0, frame_count // 2, frame_count - 1]

# Show frames
for frame_index in frames_to_preview:
    frame = reader.get_data(frame_index)
    st.image(frame, caption=f"Frame {frame_index} preview", use_column_width=True)

reader.close()

# 4️⃣ Initialize MediaPipe
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
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 5️⃣ Process video and calculate angles
left_angles, right_angles, skipped_frames = [], [], 0
left_arm_angles, right_arm_angles = [], []

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

                left_knee = calc([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
                right_knee = calc([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
                left_arm = calc([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
                right_arm = calc([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
                
                left_angles.append(left_knee)
                right_angles.append(right_knee)
                left_arm_angles.append(left_arm)
                right_arm_angles.append(right_arm)

            else:
                skipped_frames += 1

        reader.close()
    except Exception as e:
        st.error(f"❌ Error reading video frames: {e}")
        st.stop()

# 6️⃣ Frame count and skipped frames warning
frame_count = len(left_angles)  # Assuming equal length for all angles
if frame_count > 0 and skipped_frames / frame_count > 0.2:
    st.warning(f"⚠️ Warning: More than 20% of frames were skipped.")

# 7️⃣ Data frame for angle tracking
df = pd.DataFrame({
    "Frame": list(range(frame_count)),
    "Left Knee Angle": left_angles,
    "Right Knee Angle": right_angles,
    "Left Arm Angle": left_arm_angles,
    "Right Arm Angle": right_arm_angles
})

# Display DataFrame
st.write(df)

# 8️⃣ Plot the angles
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Frame"], df["Left Knee Angle"], label="Left Knee Angle", color="blue")
ax.plot(df["Frame"], df["Right Knee Angle"], label="Right Knee Angle", color="green")
ax.plot(df["Frame"], df["Left Arm Angle"], label="Left Arm Angle", color="orange")
ax.plot(df["Frame"], df["Right Arm Angle"], label="Right Arm Angle", color="red")
ax.set_xlabel("Frame")
ax.set_ylabel("Angle (°)")
ax.legend()

# Display the plot
st.pyplot(fig)
